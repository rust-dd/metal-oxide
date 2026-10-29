type Position = (usize, usize);

#[cfg(test)]
mod tests;

/// The names of the inline modules around `position`, a 1-based line and character column of
/// `text`, outermost first. Comments and literals are skipped; other braces open unnamed scopes.
pub(crate) fn enclosing(text: &str, position: Position) -> Vec<String> {
    let mut scanner = Scanner {
        characters: text.chars().collect(),
        index: 0,
        line: 1,
        column: 1,
    };
    let mut scopes = Vec::new();
    let mut recent = [Token::Other, Token::Other];
    while let Some((start, token)) = scanner.token() {
        if start >= position {
            break;
        }
        match &token {
            Token::Open => scopes.push(match &recent {
                [Token::Word(keyword, false), Token::Word(name, _)] if keyword == "mod" => {
                    Some(name.clone())
                }
                _ => None,
            }),
            Token::Close => {
                scopes.pop();
            }
            Token::Word(..) | Token::Other => {}
        }
        recent[0] = std::mem::replace(&mut recent[1], token);
    }
    scopes.into_iter().flatten().collect()
}

enum Token {
    Open,
    Close,
    /// An identifier or keyword; raw identifiers keep their name without `r#`.
    Word(String, bool),
    Other,
}

struct Scanner {
    characters: Vec<char>,
    index: usize,
    line: usize,
    column: usize,
}

impl Scanner {
    fn peek(&self, offset: usize) -> Option<char> {
        self.characters.get(self.index + offset).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek(0)?;
        self.index += 1;
        if character == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(character)
    }

    fn token(&mut self) -> Option<(Position, Token)> {
        loop {
            let start = (self.line, self.column);
            match (self.peek(0)?, self.peek(1)) {
                (character, _) if character.is_whitespace() => {
                    self.bump();
                }
                ('/', Some('/')) => {
                    while self.peek(0).is_some_and(|character| character != '\n') {
                        self.bump();
                    }
                }
                ('/', Some('*')) => self.block_comment(),
                ('{', _) => return self.single(start, Token::Open),
                ('}', _) => return self.single(start, Token::Close),
                ('"', _) => {
                    self.bump();
                    self.quoted('"');
                    return Some((start, Token::Other));
                }
                ('\'', _) => {
                    self.bump();
                    self.character_or_lifetime();
                    return Some((start, Token::Other));
                }
                (character, _) if character == '_' || character.is_alphabetic() => {
                    return Some((start, self.word()));
                }
                _ => return self.single(start, Token::Other),
            }
        }
    }

    fn single(&mut self, start: Position, token: Token) -> Option<(Position, Token)> {
        self.bump();
        Some((start, token))
    }

    fn name(&mut self) -> String {
        let mut name = String::new();
        while let Some(character) = self
            .peek(0)
            .filter(|character| *character == '_' || character.is_alphanumeric())
        {
            name.push(character);
            self.bump();
        }
        name
    }

    /// A word, or the literal that a prefix such as `r`, `b` or `c` starts.
    fn word(&mut self) -> Token {
        let name = self.name();
        match (name.as_str(), self.peek(0)) {
            ("r" | "br" | "cr", Some('"' | '#')) if self.raw_string() => Token::Other,
            ("r", Some('#')) => {
                self.bump();
                Token::Word(self.name(), true)
            }
            ("b" | "c", Some('"')) | ("b", Some('\'')) => {
                let close = self.bump().unwrap();
                self.quoted(close);
                Token::Other
            }
            _ => Token::Word(name, false),
        }
    }

    /// Skips the hashes, quotes and body of a raw string, when one starts here.
    fn raw_string(&mut self) -> bool {
        let hashes = (0..)
            .take_while(|offset| self.peek(*offset) == Some('#'))
            .count();
        if self.peek(hashes) != Some('"') {
            return false;
        }
        for _ in 0..=hashes {
            self.bump();
        }
        while let Some(character) = self.bump() {
            if character == '"' && (0..hashes).all(|offset| self.peek(offset) == Some('#')) {
                for _ in 0..hashes {
                    self.bump();
                }
                break;
            }
        }
        true
    }

    /// Skips an escaped literal's body through its closing `close`.
    fn quoted(&mut self, close: char) {
        while let Some(character) = self.bump() {
            if character == '\\' {
                self.bump();
            } else if character == close {
                break;
            }
        }
    }

    /// After a `'`: skips a character literal; a lifetime or label leaves its name to lex.
    fn character_or_lifetime(&mut self) {
        match (self.peek(0), self.peek(1)) {
            (Some('\\'), _) => self.quoted('\''),
            (Some(_), Some('\'')) => {
                self.bump();
                self.bump();
            }
            _ => {}
        }
    }

    fn block_comment(&mut self) {
        let mut depth = 0_usize;
        while let Some(character) = self.bump() {
            match (character, self.peek(0)) {
                ('/', Some('*')) => {
                    self.bump();
                    depth += 1;
                }
                ('*', Some('/')) => {
                    self.bump();
                    depth -= 1;
                    if depth == 0 {
                        return;
                    }
                }
                _ => {}
            }
        }
    }
}
