use crate::{
    BuiltKernels, Compiler, Error,
    compile::{BuildLock, Kernels},
    metadata::{self, Metadata},
    modules,
    process::{self, Result},
    recorded::{Recorded, Request},
    saved::{self, Saved},
    state::State,
    toolchain::{self, Metal},
};
use metal_oxide_artifact::{MathMode, sha256};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, PoisonError},
};

#[cfg(test)]
mod tests;

type Position = (usize, usize);

/// Lines after the kept item; appending keeps every position, and `#[kernel]` resolves in any
/// module.
const MACROS: &str = "\n#[macro_use]\nextern crate metal_oxide;\n";
const PRELUDE: &str = "#[allow(unused_imports)]\nuse metal_oxide::device::*;\n";

/// Kernels that a `#[kernel]` function or a `#[kernels]` module writes in a host crate.
///
/// The device crate's root is a copy of `file` in which every character outside
/// `start..end` is a space, so each span keeps its line and column.
pub struct SourceKernels {
    /// The host package's manifest directory.
    pub package: PathBuf,
    /// The source file as rustc names it: relative to the workspace root, where rustc runs.
    pub file: PathBuf,
    /// 1-based line and column of the first kept character.
    pub start: Position,
    /// 1-based line and column just after the last kept character.
    pub end: Position,
    /// A range inside the kept one that is blanked too, such as the `#[kernels]` attribute.
    pub hole: Option<(Position, Position)>,
    /// The item's name, which keeps one device crate per item.
    pub item: String,
    /// Appends `use metal_oxide::device::*;` for a kernel written outside a module.
    pub prelude: bool,
    pub math_mode: MathMode,
    /// Crates besides the device API that the kernels use, as code names them, from the
    /// host's dependency graph.
    pub dependencies: Vec<String>,
    /// Records line tables and sources for a GPU capture.
    pub debug_info: bool,
}

impl SourceKernels {
    /// Builds the device crate under the host's target directory, reuses the artifact of
    /// identical inputs or uses the one saved beside the package's sources.
    pub fn build(&self) -> std::result::Result<BuiltKernels, Error> {
        self.run().map_err(Error)
    }

    fn run(&self) -> Result<BuiltKernels> {
        let path = std::env::current_dir()?.join(&self.file);
        let file = path
            .strip_prefix(&self.package)
            .map_err(|_| "kernels must be written in their package's directory")?;
        let original = std::fs::read_to_string(&path)?;
        let modules = modules::enclosing(&original, self.start);
        let mut text = blank(&original, (self.start, self.end), self.hole);
        text.push_str(MACROS);
        if self.prelude {
            text.push_str(PRELUDE);
        }
        let source = serde_json::to_vec(&(file, &self.item, &text, &self.dependencies))?;
        let request = Request {
            source: sha256(&source),
            math_mode: self.math_mode,
            debug_info: self.debug_info,
        };
        let saved = Saved::new(&self.package, file, &modules, &self.item, self.math_mode);
        saved::obtain(
            &self.package,
            &saved,
            &request,
            || {
                let Ok(metal) = Metal::current() else {
                    return Ok(None);
                };
                let (directory, _) = self.directory(&modules)?;
                let state = State::read(&directory);
                Ok(
                    state
                        .and_then(|state| state.reuse(&self.package, &request, &metal.toolchain())),
                )
            },
            || {
                toolchain::nightly()?;
                let metal = Metal::current()?;
                let host = host(&self.package, true)?;
                let compiler = Compiler::search(&host)?;
                Ok((host, compiler, metal))
            },
            |(host, compiler, metal)| {
                self.compile(&text, &request, &modules, host, compiler, metal)
            },
        )
    }

    /// The directory of the device crate of the item in the file's inline `modules`, and the
    /// key that names it.
    fn directory(&self, modules: &[String]) -> Result<(PathBuf, String)> {
        let identity = serde_json::to_vec(&(
            &self.package,
            &self.file,
            modules,
            &self.item,
            self.math_mode,
            self.debug_info,
        ))?;
        let key = sha256(&identity)[..16].to_owned();
        let workspace = host(&self.package, false)?;
        let directory = workspace.target_directory.join("metal/sources").join(&key);
        Ok((directory, key))
    }

    fn compile(
        &self,
        text: &str,
        request: &Request,
        modules: &[String],
        host: Arc<Metadata>,
        compiler: Compiler,
        metal: &'static Metal,
    ) -> Result<(BuiltKernels, Recorded)> {
        let (directory, key) = self.directory(modules)?;
        // The expansions of one item for several targets share its directory; the lock keeps
        // one from rewriting it while another builds from it.
        let lock = BuildLock::acquire(&host.target_directory)?;
        write(&directory.join(&self.file), text)?;
        let config = format!(
            "[build]\ntarget-dir = {:?}\n",
            host.target_directory.display().to_string()
        );
        write(&directory.join(".cargo/config.toml"), &config)?;
        write(&directory.join("Cargo.toml"), &self.manifest(&host, &key)?)?;
        // The host's lock file pins the versions of the packages that the device crate shares.
        std::fs::copy(
            host.workspace_root.join("Cargo.lock"),
            directory.join("Cargo.lock"),
        )?;
        process::capture(metadata::cargo().current_dir(&directory).args([
            "update",
            "--workspace",
            "--offline",
            "--quiet",
        ]))?;
        let project = metadata::load(&directory.join("Cargo.toml"))?;
        let debug_info = self.debug_info.then(|| host.workspace_root.clone());
        Kernels::new(
            project,
            compiler,
            metal,
            self.math_mode,
            Some(host),
            debug_info,
        )
        .record(&self.package, request.clone(), &directory, &lock)
    }

    fn manifest(&self, host: &Metadata, key: &str) -> Result<String> {
        let package = host
            .packages
            .iter()
            .find(|package| package.manifest_path == self.package.join("Cargo.toml"))
            .ok_or("the kernels' package is missing from its Cargo metadata")?;
        let mut manifest = format!(
            "[package]\nname = \"metal-oxide-kernels-{key}\"\nversion = \"0.0.0\"\nedition = {:?}\npublish = false\nautobins = false\nautoexamples = false\nautotests = false\nautobenches = false\n\n[lib]\npath = {:?}\n\n[dependencies]\n",
            package.edition,
            self.file.display().to_string()
        );
        let device = source(host, "metal-oxide-device")?;
        manifest.push_str(&format!(
            "metal_oxide = {{ package = \"metal-oxide-device\", path = {:?} }}\n",
            device.display().to_string()
        ));
        for name in &self.dependencies {
            let declaration = package.declaration(name)?;
            let path = source(host, &declaration.package)?;
            manifest.push_str(&format!(
                "{name} = {{ package = {:?}, path = {:?}, default-features = {}, features = {:?} }}\n",
                declaration.package,
                path.display().to_string(),
                declaration.default_features,
                declaration.features
            ));
        }
        manifest.push_str("\n[workspace]\n");
        Ok(manifest)
    }
}

/// The host package's Cargo metadata, read once per compiler process; `dependencies` resolves
/// the packages beyond the workspace members.
pub(crate) fn host(package: &Path, dependencies: bool) -> Result<Arc<Metadata>> {
    type Hosts = HashMap<(PathBuf, bool), Arc<Metadata>>;
    static HOSTS: Mutex<Option<Hosts>> = Mutex::new(None);
    let mut hosts = HOSTS.lock().unwrap_or_else(PoisonError::into_inner);
    let hosts = hosts.get_or_insert_with(HashMap::new);
    let key = (package.to_owned(), dependencies);
    if let Some(metadata) = hosts.get(&key) {
        return Ok(Arc::clone(metadata));
    }
    let metadata = Arc::new(metadata::workspace(
        &package.join("Cargo.toml"),
        dependencies,
    )?);
    hosts.insert(key, Arc::clone(&metadata));
    Ok(metadata)
}

/// The source directory of the one package named `name` in the host's dependency graph.
fn source(host: &Metadata, name: &str) -> Result<PathBuf> {
    let mut found = host.packages.iter().filter(|package| package.name == name);
    match (found.next(), found.next()) {
        (Some(package), None) => Ok(package.manifest_path.parent().unwrap().to_owned()),
        (None, _) => {
            Err(format!("the kernels use {name}, which the host does not depend on").into())
        }
        (Some(_), Some(_)) => Err(format!("the host depends on several versions of {name}").into()),
    }
}

type Range = (Position, Position);

/// The `kept` range of `text` at its original lines and columns, with `hole` blanked: the
/// lines before it are empty, the first one starts with spaces and the lines after it are
/// left out, so edits outside it change nothing unless they move it.
fn blank(text: &str, kept: Range, hole: Option<Range>) -> String {
    let inside = |position, (start, end): Range| position >= start && position < end;
    let mut output = String::with_capacity(text.len());
    for (index, line) in text.split_inclusive('\n').enumerate() {
        let number = index + 1;
        if number < kept.0.0 {
            output.push('\n');
            continue;
        }
        if number > kept.1.0 {
            break;
        }
        for (column, character) in line.chars().enumerate() {
            let position = (number, column + 1);
            if position >= kept.1 {
                break;
            }
            let shown = inside(position, kept) && !hole.is_some_and(|hole| inside(position, hole));
            output.push(if shown || matches!(character, '\n' | '\r') {
                character
            } else {
                ' '
            });
        }
    }
    output.push('\n');
    output
}

/// Writes `text` unless the file already holds it; returns whether it wrote.
fn write(path: &Path, text: &str) -> Result<bool> {
    if std::fs::read_to_string(path).ok().as_deref() == Some(text) {
        return Ok(false);
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(path, text)?;
    Ok(true)
}
