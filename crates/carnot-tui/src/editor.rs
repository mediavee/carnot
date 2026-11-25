//! Responsive multi-line terminal prompt editor with history, cursor navigation,
//! word deletion, bracketed paste, and visual line wrapping.

use unicode_width::UnicodeWidthChar;

#[derive(Debug, Default)]
pub(crate) struct PromptEditor {
    chars: Vec<char>,
    /// Cursor position in character indices (`0..=chars.len()`).
    cursor: usize,
    history: Vec<String>,
    /// `None` when editing fresh input; `Some(idx)` when browsing history.
    history_idx: Option<usize>,
    /// Draft saved when first stepping up into history.
    saved_draft: String,
    /// Vertical scroll offset inside the visible input viewport.
    scroll_row: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct RenderedEditor {
    pub(crate) lines: Vec<String>,
    pub(crate) cursor_col: u16,
    pub(crate) cursor_row: u16,
    pub(crate) total_rows: usize,
}

impl PromptEditor {
    pub(crate) fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    pub(crate) fn text(&self) -> String {
        self.chars.iter().collect()
    }

    pub(crate) fn set_text(&mut self, s: &str) {
        self.chars = s.chars().collect();
        self.cursor = self.chars.len();
        self.history_idx = None;
        self.scroll_row = 0;
    }

    pub(crate) fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
        self.history_idx = None;
        self.scroll_row = 0;
    }

    pub(crate) fn take_and_record(&mut self) -> String {
        let out = self.text();
        let trimmed = out.trim().to_string();
        if !trimmed.is_empty() && self.history.last() != Some(&trimmed) {
            self.history.push(trimmed);
            if self.history.len() > 200 {
                self.history.remove(0);
            }
        }
        self.clear();
        self.saved_draft.clear();
        out
    }

    pub(crate) fn insert_char(&mut self, ch: char) {
        self.history_idx = None;
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    pub(crate) fn insert_str(&mut self, s: &str) {
        self.history_idx = None;
        let normalized = s.replace("\r\n", "\n").replace('\r', "\n");
        for ch in normalized.chars() {
            self.chars.insert(self.cursor, ch);
            self.cursor += 1;
        }
    }

    pub(crate) fn backspace(&mut self) {
        if self.cursor > 0 {
            self.history_idx = None;
            self.cursor -= 1;
            self.chars.remove(self.cursor);
        }
    }

    pub(crate) fn delete(&mut self) {
        if self.cursor < self.chars.len() {
            self.history_idx = None;
            self.chars.remove(self.cursor);
        }
    }

    pub(crate) fn delete_word_back(&mut self) {
        if self.cursor == 0 {
            return;
        }
        self.history_idx = None;
        let mut new_cur = self.cursor;
        while new_cur > 0 && self.chars[new_cur - 1].is_whitespace() {
            new_cur -= 1;
        }
        while new_cur > 0 && !self.chars[new_cur - 1].is_whitespace() {
            new_cur -= 1;
        }
        self.chars.drain(new_cur..self.cursor);
        self.cursor = new_cur;
    }

    pub(crate) fn kill_to_line_start(&mut self) {
        let start = self.line_start(self.cursor);
        if start < self.cursor {
            self.history_idx = None;
            self.chars.drain(start..self.cursor);
            self.cursor = start;
        }
    }

    pub(crate) fn kill_to_line_end(&mut self) {
        let end = self.line_end(self.cursor);
        if self.cursor < end {
            self.history_idx = None;
            self.chars.drain(self.cursor..end);
        }
    }

    pub(crate) fn move_left(&mut self) {
        self.cursor = self.cursor.saturating_sub(1);
    }

    pub(crate) fn move_right(&mut self) {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
        }
    }

    pub(crate) fn move_word_left(&mut self) {
        while self.cursor > 0 && self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
    }

    pub(crate) fn move_word_right(&mut self) {
        let len = self.chars.len();
        while self.cursor < len && !self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
        while self.cursor < len && self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
    }

    pub(crate) fn move_home(&mut self) {
        self.cursor = self.line_start(self.cursor);
    }

    pub(crate) fn move_end(&mut self) {
        self.cursor = self.line_end(self.cursor);
    }

    /// Move cursor up one logical line, or step back into prompt history when on the first line.
    pub(crate) fn move_up_or_history(&mut self) {
        let cur_start = self.line_start(self.cursor);
        if cur_start == 0 {
            self.history_prev();
            return;
        }
        let col = self.cursor - cur_start;
        let prev_end = cur_start - 1;
        let prev_start = self.line_start(prev_end);
        let prev_len = prev_end - prev_start;
        self.cursor = prev_start + col.min(prev_len);
    }

    /// Move cursor down one logical line, or step forward in history when on the last line.
    pub(crate) fn move_down_or_history(&mut self) {
        let cur_end = self.line_end(self.cursor);
        if cur_end >= self.chars.len() {
            self.history_next();
            return;
        }
        let cur_start = self.line_start(self.cursor);
        let col = self.cursor - cur_start;
        let next_start = cur_end + 1;
        let next_end = self.line_end(next_start);
        let next_len = next_end - next_start;
        self.cursor = next_start + col.min(next_len);
    }

    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next_idx = match self.history_idx {
            None => {
                self.saved_draft = self.text();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_idx = Some(next_idx);
        self.chars = self.history[next_idx].chars().collect();
        self.cursor = self.chars.len();
    }

    fn history_next(&mut self) {
        let Some(cur) = self.history_idx else {
            return;
        };
        if cur + 1 < self.history.len() {
            let next_idx = cur + 1;
            self.history_idx = Some(next_idx);
            self.chars = self.history[next_idx].chars().collect();
            self.cursor = self.chars.len();
        } else {
            self.history_idx = None;
            self.chars = self.saved_draft.chars().collect();
            self.cursor = self.chars.len();
        }
    }

    fn line_start(&self, pos: usize) -> usize {
        let mut p = pos.min(self.chars.len());
        while p > 0 && self.chars[p - 1] != '\n' {
            p -= 1;
        }
        p
    }

    fn line_end(&self, pos: usize) -> usize {
        let mut p = pos.min(self.chars.len());
        while p < self.chars.len() && self.chars[p] != '\n' {
            p += 1;
        }
        p
    }

    /// Wrap text for a viewport of `inner_width` x `max_visible_rows`, returning visible
    /// lines and the cursor `(col, row)` relative to the inner viewport.
    pub(crate) fn render(&mut self, inner_width: usize, max_visible_rows: usize) -> RenderedEditor {
        let w = inner_width.max(4);
        let mut rows: Vec<String> = vec![String::new()];
        let mut cur_w = 0usize;
        let mut cursor_row = 0usize;
        let mut cursor_col = 0usize;

        for (idx, &ch) in self.chars.iter().enumerate() {
            if idx == self.cursor {
                cursor_row = rows.len() - 1;
                cursor_col = cur_w;
            }
            if ch == '\n' {
                rows.push(String::new());
                cur_w = 0;
                continue;
            }
            let cw = ch.width().unwrap_or(1).max(1);
            if cur_w + cw > w && cur_w > 0 {
                rows.push(String::new());
                cur_w = 0;
            }
            if let Some(last) = rows.last_mut() {
                last.push(ch);
            }
            cur_w += cw;
        }
        if self.cursor >= self.chars.len() {
            cursor_row = rows.len() - 1;
            cursor_col = cur_w;
        }

        let total_rows = rows.len();
        let vis = max_visible_rows.max(1);
        if cursor_row < self.scroll_row {
            self.scroll_row = cursor_row;
        } else if cursor_row >= self.scroll_row + vis {
            self.scroll_row = cursor_row + 1 - vis;
        }
        let max_scroll = total_rows.saturating_sub(vis);
        if self.scroll_row > max_scroll {
            self.scroll_row = max_scroll;
        }

        let end = (self.scroll_row + vis).min(total_rows);
        let visible_lines = rows[self.scroll_row..end].to_vec();
        let rel_row = cursor_row.saturating_sub(self.scroll_row);

        RenderedEditor {
            lines: visible_lines,
            cursor_col: cursor_col.min(w.saturating_sub(1)) as u16,
            cursor_row: rel_row.min(vis.saturating_sub(1)) as u16,
            total_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiline_edit_and_history_work() {
        let mut ed = PromptEditor::default();
        ed.insert_str("first prompt");
        assert_eq!(ed.take_and_record(), "first prompt");

        ed.insert_str("draft");
        ed.move_up_or_history();
        assert_eq!(ed.text(), "first prompt");
        ed.move_down_or_history();
        assert_eq!(ed.text(), "draft");

        ed.insert_char('\n');
        ed.insert_str("line two");
        ed.delete_word_back();
        assert_eq!(ed.text(), "draft\nline ");
        let r = ed.render(40, 4);
        assert_eq!(r.total_rows, 2);
        assert_eq!(r.cursor_row, 1);
    }
}
