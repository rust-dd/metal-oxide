use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::{Compiler, Config};
use rustc_middle::ty::TyCtxt;
use rustc_session::config::PrintKind;
use rustc_span::{BytePos, Span};

pub(crate) struct Frontend {
    pub(crate) output: Option<std::path::PathBuf>,
    pub(crate) math_mode: metal_oxide_artifact::MathMode,
    pub(crate) help: bool,
    pub(crate) arguments: Vec<String>,
}

impl Callbacks for Frontend {
    fn config(&mut self, config: &mut Config) {
        if config.opts.unstable_opts.unpretty.is_some() {
            self.output = None;
        }
        if !self.help
            && !config.opts.describe_lints
            && config.opts.prints.iter().all(|request| {
                matches!(
                    request.kind,
                    PrintKind::NativeStaticLibs | PrintKind::LinkArgs
                )
            })
            && let Some(directory) = self.output.clone()
        {
            config.psess_created = Some(Box::new(move |session| {
                if let Err(error) = crate::output::prepare(&directory) {
                    session.dcx().fatal(error.to_string());
                }
            }));
        }
        config.make_codegen_backend = Some(Box::new(|_| Box::new(crate::backend::FrontendBackend)));
        config.opts.unstable_opts.crate_attr.extend([
            "feature(register_tool)".into(),
            "register_tool(metal_oxide_compiler)".into(),
            "allow(internal_features)".into(),
        ]);
        config.opts.unstable_opts.mir_opt_level = Some(0);
        // Core's unsafe-precondition checks compare raw addresses and abort without unwinding,
        // which devices cannot do; release builds disable them too.
        config.opts.unstable_opts.ub_checks = Some(false);
        config.opts.unstable_opts.always_encode_mir = true;
        config.opts.unstable_opts.enforce_type_length_limit = true;
    }

    fn after_analysis<'tcx>(&mut self, _compiler: &Compiler, tcx: TyCtxt<'tcx>) -> Compilation {
        if tcx.sess.opts.unstable_opts.unpretty.is_some() {
            return Compilation::Continue;
        }
        crate::target::validate(tcx);
        let entries = crate::collect::kernels(tcx);
        tcx.dcx().abort_if_errors();
        if let Some(directory) = &self.output {
            match crate::import::ModuleImporter::new(tcx).import(&entries) {
                Ok(module) => {
                    if let Err(error) = crate::output::write(
                        directory,
                        &module,
                        tcx,
                        &entries,
                        &self.arguments,
                        self.math_mode,
                    ) {
                        match error.downcast_ref::<metal_oxide_ir::Error>() {
                            Some(error) => report(tcx, error),
                            None => {
                                tcx.dcx().err(error.to_string());
                            }
                        }
                    }
                }
                Err(rejection) => {
                    let error = tcx.dcx().struct_span_err(rejection.span, rejection.message);
                    match rejection.reached {
                        Some(inner) => error.with_span_note(inner, crate::import::REACHED),
                        None => error,
                    }
                    .emit();
                }
            }
            tcx.dcx().abort_if_errors();
        }
        Compilation::Continue
    }
}

/// Reports an IR error in the kernel crate: at its own location there, or at the innermost
/// kernel crate call that reached it in another crate's code.
fn report(tcx: TyCtxt<'_>, error: &metal_oxide_ir::Error) {
    let local = |span: &Span| crate::import::local(tcx, *span);
    let source = span(tcx, &error.source);
    let call = error
        .calls
        .iter()
        .filter_map(|call| span(tcx, call))
        .find(local);
    let message = error.message.clone();
    match (source, call) {
        (Some(source), _) if local(&source) => {
            tcx.dcx().span_err(source, message);
        }
        (Some(source), Some(call)) => {
            tcx.dcx()
                .struct_span_err(call, message)
                .with_span_note(source, crate::import::REACHED)
                .emit();
        }
        (None, Some(call)) => {
            let source = &error.source;
            let note = format!(
                "{} at {}:{}:{}",
                crate::import::REACHED,
                source.file,
                source.line,
                source.column
            );
            tcx.dcx()
                .struct_span_err(call, message)
                .with_note(note)
                .emit();
        }
        (Some(source), None) => {
            tcx.dcx().span_err(source, message);
        }
        (None, None) => {
            tcx.dcx().err(error.to_string());
        }
    }
}

/// The position an IR source location names in a file of this session.
fn span(tcx: TyCtxt<'_>, location: &metal_oxide_ir::SourceLocation) -> Option<Span> {
    let source_map = tcx.sess.source_map();
    let file = source_map
        .files()
        .iter()
        .find(|file| file.name.prefer_remapped_unconditionally().to_string() == location.file)
        .cloned()?;
    let line = (location.line as usize)
        .checked_sub(1)
        .filter(|line| *line < file.lines().len())?;
    source_map.ensure_source_file_source_present(&file);
    let column = file.get_line(line).map_or(0, |text| {
        text.chars()
            .take(location.column.saturating_sub(1) as usize)
            .map(char::len_utf8)
            .sum::<usize>()
    });
    let start = file.line_bounds(line).start + BytePos(column as u32);
    Some(Span::with_root_ctxt(start, start))
}
