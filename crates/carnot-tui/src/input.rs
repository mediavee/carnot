//! Keyboard, mouse, scroll, and interactive modal input handling for `TuiApp`.

use crate::markdown;
use crate::state::{NoticeKind, PermissionReply, TuiApp};
use carnot::UserQuestionAnswer;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEvent, MouseEventKind};
use std::time::{Duration, Instant};

impl TuiApp {
    pub(crate) fn scroll_up(&mut self, lines: usize) {
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        if self.auto_follow {
            self.scroll_offset = max_scroll;
            self.auto_follow = false;
        }
        self.scroll_offset = self.scroll_offset.saturating_sub(lines);
    }

    pub(crate) fn scroll_down(&mut self, lines: usize) {
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        self.scroll_offset = (self.scroll_offset + lines).min(max_scroll);
        if self.scroll_offset >= max_scroll {
            self.auto_follow = true;
        }
    }

    pub(crate) fn jump_bottom(&mut self) {
        self.auto_follow = true;
        let max_scroll = self
            .cached_lines
            .len()
            .saturating_sub(self.last_viewport_height);
        self.scroll_offset = max_scroll;
    }

    pub(crate) fn handle_mouse(&mut self, mouse: MouseEvent) {
        match mouse.kind {
            MouseEventKind::ScrollUp => self.scroll_up(3),
            MouseEventKind::ScrollDown => self.scroll_down(3),
            _ => {}
        }
    }

    /// Returns `true` if the application should exit.
    pub(crate) async fn handle_key(&mut self, key: KeyEvent) -> bool {
        if key.kind != KeyEventKind::Press {
            return false;
        }

        // Priority 1: Tool Approval Modal
        if let Some(perm) = self.active_perm.take() {
            match key.code {
                KeyCode::Char('y' | 'Y') => {
                    let _ = perm.reply.send(PermissionReply::AllowOnce);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('a' | 'A') => {
                    let _ = perm.reply.send(PermissionReply::AllowAlways);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('n' | 'N') | KeyCode::Esc => {
                    let _ = perm.reply.send(PermissionReply::Deny);
                    self.active_perm = self.perm_queue.pop_front();
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    let _ = perm.reply.send(PermissionReply::Deny);
                    self.active_perm = self.perm_queue.pop_front();
                    self.agent.abort().await;
                }
                _ => {
                    self.active_perm = Some(perm);
                }
            }
            return false;
        }

        // Priority 2: AskUser Interactive Modal
        if self.active_question.is_some() {
            self.handle_question_key(key);
            return false;
        }

        // Priority 3: Interactive Model Picker
        if self.model_picker.is_some() {
            self.handle_model_picker_key(key).await;
            return false;
        }

        // Global Ctrl shortcuts
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    if self.running {
                        self.agent.abort().await;
                        self.push_notice(NoticeKind::Warning, "Aborting current turn…", None);
                        return false;
                    }
                    if !self.editor.is_empty() {
                        self.editor.clear();
                        self.last_ctrl_c = Some(Instant::now());
                        return false;
                    }
                    if let Some(prev) = self.last_ctrl_c
                        && prev.elapsed() < Duration::from_millis(1600)
                    {
                        return true;
                    }
                    self.last_ctrl_c = Some(Instant::now());
                    return false;
                }
                KeyCode::Char('d') if self.editor.is_empty() && !self.running => {
                    return true;
                }
                KeyCode::Char('t') => {
                    self.show_thinking = !self.show_thinking;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('o') => {
                    self.expand_tools = !self.expand_tools;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('p') if !self.running => {
                    self.open_model_picker();
                    return false;
                }
                KeyCode::Char('l') => {
                    self.items.clear();
                    self.scroll_offset = 0;
                    self.auto_follow = true;
                    self.cache_dirty = true;
                    return false;
                }
                KeyCode::Char('j') => {
                    self.editor.insert_char('\n');
                    return false;
                }
                KeyCode::Char('a') => {
                    self.editor.move_home();
                    return false;
                }
                KeyCode::Char('e') => {
                    self.editor.move_end();
                    return false;
                }
                KeyCode::Char('w') => {
                    self.editor.delete_word_back();
                    return false;
                }
                KeyCode::Char('u') => {
                    if self.editor.is_empty() {
                        self.scroll_up(self.last_viewport_height / 2);
                    } else {
                        self.editor.kill_to_line_start();
                    }
                    return false;
                }
                KeyCode::Char('k') => {
                    self.editor.kill_to_line_end();
                    return false;
                }
                KeyCode::Char('d') if !self.editor.is_empty() => {
                    self.editor.delete();
                    return false;
                }
                _ => {}
            }
        }

        // Scroll shortcuts
        match key.code {
            KeyCode::PageUp => {
                self.scroll_up(self.last_viewport_height.saturating_sub(2).max(4));
                return false;
            }
            KeyCode::PageDown => {
                self.scroll_down(self.last_viewport_height.saturating_sub(2).max(4));
                return false;
            }
            KeyCode::End if self.editor.is_empty() => {
                self.jump_bottom();
                return false;
            }
            KeyCode::Esc => {
                if self.running {
                    self.agent.abort().await;
                    self.push_notice(NoticeKind::Warning, "Aborting current turn…", None);
                } else if !self.editor.is_empty() {
                    self.editor.clear();
                }
                return false;
            }
            _ => {}
        }

        // Slash autocomplete navigation when active
        let slash = self.slash_matches();
        if !slash.is_empty() {
            match key.code {
                KeyCode::Up => {
                    self.slash_selected = self
                        .slash_selected
                        .checked_sub(1)
                        .unwrap_or(slash.len() - 1);
                    return false;
                }
                KeyCode::Down => {
                    self.slash_selected = (self.slash_selected + 1) % slash.len();
                    return false;
                }
                KeyCode::Tab => {
                    let chosen = slash[self.slash_selected.min(slash.len() - 1)];
                    let needs_arg = matches!(
                        chosen.name,
                        "model" | "effort" | "cache" | "approve" | "compact"
                    );
                    if needs_arg {
                        self.editor.set_text(&format!("/{} ", chosen.name));
                    } else {
                        self.editor.set_text(&format!("/{}", chosen.name));
                    }
                    self.slash_selected = 0;
                    return false;
                }
                KeyCode::Enter
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::SHIFT | KeyModifiers::ALT) =>
                {
                    let chosen = slash[self.slash_selected.min(slash.len() - 1)];
                    self.editor.clear();
                    self.slash_selected = 0;
                    return self.handle_slash_command(chosen.name).await;
                }
                _ => {}
            }
        }

        // Standard composer editing
        match key.code {
            KeyCode::Enter => {
                let is_shift = key.modifiers.contains(KeyModifiers::SHIFT);
                let is_alt = key.modifiers.contains(KeyModifiers::ALT);
                if is_shift || (is_alt && !self.running) {
                    self.editor.insert_char('\n');
                } else {
                    let raw = self.editor.take_and_record();
                    let trimmed = raw.trim();
                    if trimmed.is_empty() {
                        return false;
                    }
                    if let Some(cmd) = trimmed.strip_prefix('/') {
                        return self.handle_slash_command(cmd).await;
                    }
                    if self.compacting {
                        self.push_notice(
                            NoticeKind::Warning,
                            "Compaction in progress — wait a moment before sending",
                            None,
                        );
                        self.editor.set_text(&raw);
                    } else if self.running {
                        let preview = markdown::truncate_display(&trimmed.replace('\n', " "), 80);
                        if is_alt {
                            self.agent.follow_up(trimmed.to_string()).await;
                            self.push_notice(
                                NoticeKind::Info,
                                format!("Follow-up queued → {preview}"),
                                None,
                            );
                        } else {
                            self.agent.steer(trimmed.to_string()).await;
                            self.push_notice(
                                NoticeKind::Info,
                                format!("Steer queued → {preview}"),
                                None,
                            );
                        }
                    } else {
                        self.start_prompt(trimmed.to_string());
                    }
                }
            }
            KeyCode::Backspace => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.delete_word_back();
                } else {
                    self.editor.backspace();
                }
                self.slash_selected = 0;
            }
            KeyCode::Delete => self.editor.delete(),
            KeyCode::Left => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.move_word_left();
                } else {
                    self.editor.move_left();
                }
            }
            KeyCode::Right => {
                if key.modifiers.contains(KeyModifiers::ALT) {
                    self.editor.move_word_right();
                } else {
                    self.editor.move_right();
                }
            }
            KeyCode::Home => self.editor.move_home(),
            KeyCode::End => self.editor.move_end(),
            KeyCode::Up => self.editor.move_up_or_history(),
            KeyCode::Down => self.editor.move_down_or_history(),
            KeyCode::Char(c) => {
                self.editor.insert_char(c);
                self.slash_selected = 0;
            }
            _ => {}
        }
        false
    }

    pub(crate) fn handle_question_key(&mut self, key: KeyEvent) {
        let Some(mut pq) = self.active_question.take() else {
            return;
        };
        let total_rows = pq.total_rows();
        let is_multi = pq.current().is_some_and(|q| q.multi_select);

        match key.code {
            KeyCode::Up
                if !pq.typing_custom || pq.current().is_some_and(|q| !q.options.is_empty()) =>
            {
                pq.selected_row = pq.selected_row.checked_sub(1).unwrap_or(total_rows - 1);
                pq.typing_custom = pq.is_custom_row();
                self.active_question = Some(pq);
            }
            KeyCode::Down
                if !pq.typing_custom || pq.current().is_some_and(|q| !q.options.is_empty()) =>
            {
                pq.selected_row = (pq.selected_row + 1) % total_rows;
                pq.typing_custom = pq.is_custom_row();
                self.active_question = Some(pq);
            }
            KeyCode::Char(' ') if is_multi && !pq.is_custom_row() => {
                if !pq.checked.remove(&pq.selected_row) {
                    pq.checked.insert(pq.selected_row);
                }
                self.active_question = Some(pq);
            }
            KeyCode::Backspace if pq.typing_custom => {
                pq.custom_input.pop();
                self.active_question = Some(pq);
            }
            KeyCode::Char(c) if pq.typing_custom => {
                pq.custom_input.push(c);
                self.active_question = Some(pq);
            }
            KeyCode::Char(c) if !pq.typing_custom && c.is_ascii_digit() => {
                if let Some(d) = c.to_digit(10)
                    && d >= 1
                    && let Some(q) = pq.current()
                    && (d as usize) <= q.options.len()
                {
                    pq.selected_row = (d as usize) - 1;
                    if is_multi && !pq.checked.remove(&pq.selected_row) {
                        pq.checked.insert(pq.selected_row);
                    }
                }
                self.active_question = Some(pq);
            }
            KeyCode::Enter => {
                if let Some(q) = pq.current().cloned() {
                    let mut selected = Vec::new();
                    let mut text = None;
                    if pq.is_custom_row() {
                        let t = pq.custom_input.trim().to_string();
                        if !t.is_empty() {
                            text = Some(t);
                        }
                    } else if q.multi_select {
                        for (idx, opt) in q.options.iter().enumerate() {
                            if pq.checked.contains(&idx) {
                                selected.push(opt.label.clone());
                            }
                        }
                        if selected.is_empty()
                            && let Some(opt) = q.options.get(pq.selected_row)
                        {
                            selected.push(opt.label.clone());
                        }
                    } else if let Some(opt) = q.options.get(pq.selected_row) {
                        selected.push(opt.label.clone());
                    }
                    pq.answers.push(UserQuestionAnswer {
                        id: q.id,
                        selected,
                        text,
                    });
                }

                pq.current_idx += 1;
                if pq.current_idx >= pq.questions.len() {
                    let _ = pq.reply.send(pq.answers);
                    self.active_question = self.question_queue.pop_front();
                } else {
                    pq.reset_step();
                    self.active_question = Some(pq);
                }
            }
            KeyCode::Esc => {
                // Return answers collected so far (empty for remaining)
                for q in pq.questions.iter().skip(pq.current_idx) {
                    pq.answers.push(UserQuestionAnswer {
                        id: q.id.clone(),
                        selected: Vec::new(),
                        text: None,
                    });
                }
                let _ = pq.reply.send(pq.answers);
                self.active_question = self.question_queue.pop_front();
            }
            _ => {
                self.active_question = Some(pq);
            }
        }
    }

    pub(crate) async fn handle_model_picker_key(&mut self, key: KeyEvent) {
        let Some(mut picker) = self.model_picker.take() else {
            return;
        };
        let count = picker.filtered().len();

        match key.code {
            KeyCode::Esc => {
                // Close without switching
            }
            KeyCode::Up if count > 0 => {
                picker.selected = picker.selected.checked_sub(1).unwrap_or(count - 1);
                self.model_picker = Some(picker);
            }
            KeyCode::Down if count > 0 => {
                picker.selected = (picker.selected + 1) % count;
                self.model_picker = Some(picker);
            }
            KeyCode::Enter => {
                let chosen = picker
                    .filtered()
                    .get(picker.selected.min(count.saturating_sub(1)))
                    .map(|m| (m.id.clone(), m.provider.clone()));
                if let Some((id, prov)) = chosen {
                    self.switch_model(&id, Some(&prov)).await;
                }
            }
            KeyCode::Backspace => {
                picker.filter.pop();
                picker.selected = 0;
                self.model_picker = Some(picker);
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                picker.filter.push(c);
                picker.selected = 0;
                self.model_picker = Some(picker);
            }
            _ => {
                self.model_picker = Some(picker);
            }
        }
    }
}
