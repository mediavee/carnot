//! Ratatui layout, cached transcript rendering, tool cards, modals, composer, and status bar.

use crate::editor;
use crate::format::{
    clip_one_line, count_diff_stats, fmt_duration_ms, fmt_tokens, render_diff_preview,
};
use crate::markdown::{display_width, render_markdown, truncate_display, wrap_spans};
use crate::state::{NoticeKind, SPINNER, SlashSpec, ToolStatus, TranscriptItem, TuiApp};
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use std::time::Duration;

impl TuiApp {
    pub(crate) fn rebuild_transcript_cache(&mut self, width: usize) {
        let content_w = width.saturating_sub(2).max(20);
        let p = &self.palette;
        let spin = SPINNER[self.spinner_frame % SPINNER.len()];
        let mut lines: Vec<Line<'static>> = Vec::new();

        for item in &self.items {
            if !lines.is_empty() {
                lines.push(Line::default());
            }
            match item {
                TranscriptItem::User { text } => {
                    let bar = Span::styled("┃ ", p.accent_bold());
                    let msg_style = if p.color {
                        Style::default().fg(p.fg).bg(p.surface)
                    } else {
                        Style::default().add_modifier(Modifier::BOLD)
                    };
                    for raw_line in text.lines() {
                        let spans = vec![Span::styled(raw_line.to_string(), msg_style)];
                        wrap_spans(
                            &spans,
                            content_w,
                            vec![bar.clone()],
                            std::slice::from_ref(&bar),
                            &mut lines,
                        );
                    }
                }
                TranscriptItem::Assistant { text } => {
                    let md_lines = render_markdown(text, content_w, p);
                    for l in md_lines {
                        let mut prefixed = vec![Span::raw("  ")];
                        prefixed.extend(l.spans);
                        lines.push(Line::from(prefixed));
                    }
                }
                TranscriptItem::Thinking {
                    text,
                    started_at,
                    duration_ms,
                } => {
                    let ms = duration_ms.unwrap_or_else(|| started_at.elapsed().as_millis() as u64);
                    let secs = format!("{:.1}s", ms as f64 / 1000.0);
                    let active = duration_ms.is_none();
                    if self.show_thinking {
                        let icon = if active { spin } else { "◇" };
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {icon} Thinking "), p.thinking()),
                            Span::styled(format!("({secs})"), p.muted()),
                        ]));
                        for raw_line in text.lines() {
                            let spans = vec![Span::styled(raw_line.to_string(), p.thinking())];
                            let pref = vec![Span::styled("  │ ", p.border())];
                            wrap_spans(&spans, content_w, pref.clone(), &pref, &mut lines);
                        }
                    } else if active {
                        lines.push(Line::from(vec![
                            Span::styled(format!("  {spin} Thinking… "), p.thinking()),
                            Span::styled(format!("({secs})  ^T to expand"), p.muted()),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::styled("  ◇ ", p.thinking()),
                            Span::styled(format!("Thought for {secs}  (^T to show)"), p.muted()),
                        ]));
                    }
                }
                TranscriptItem::Tool {
                    name,
                    summary,
                    status,
                    started_at,
                    duration_ms,
                    progress,
                    output_preview,
                    edit_diff,
                    ..
                } => {
                    let (icon, icon_style, dur_text) = match status {
                        ToolStatus::Running => {
                            let ms = started_at.elapsed().as_millis();
                            (spin, p.warning(), format!("{:.1}s", ms as f64 / 1000.0))
                        }
                        ToolStatus::Success => (
                            "✓",
                            p.success(),
                            duration_ms.map(fmt_duration_ms).unwrap_or_default(),
                        ),
                        ToolStatus::Error => (
                            "✗",
                            p.danger(),
                            duration_ms.map(fmt_duration_ms).unwrap_or_default(),
                        ),
                    };

                    let mut header_spans = vec![
                        Span::raw("  "),
                        Span::styled(icon.to_string(), icon_style),
                        Span::raw(" "),
                        Span::styled(
                            truncate_display(summary, content_w.saturating_sub(24)),
                            if *status == ToolStatus::Error {
                                p.danger()
                            } else {
                                p.text()
                            },
                        ),
                    ];
                    if let Some((old_s, new_s)) = edit_diff {
                        let (adds, dels) = count_diff_stats(old_s, new_s);
                        if adds > 0 || dels > 0 {
                            header_spans.push(Span::raw("  "));
                            if adds > 0 {
                                header_spans.push(Span::styled(format!("+{adds}"), p.success()));
                            }
                            if adds > 0 && dels > 0 {
                                header_spans.push(Span::raw(" "));
                            }
                            if dels > 0 {
                                header_spans.push(Span::styled(format!("-{dels}"), p.danger()));
                            }
                        }
                    }
                    if !dur_text.is_empty() {
                        header_spans.push(Span::styled(format!("  {dur_text}"), p.muted()));
                    }
                    lines.push(Line::from(header_spans));

                    if let Some(prog) = progress {
                        lines.push(Line::from(vec![
                            Span::styled("    ↳ ", p.border()),
                            Span::styled(prog.clone(), p.muted()),
                        ]));
                    }

                    if *status == ToolStatus::Error && !output_preview.is_empty() {
                        let err_line = clip_one_line(output_preview, content_w.saturating_sub(8));
                        lines.push(Line::from(vec![
                            Span::styled("    ↳ ", p.danger()),
                            Span::styled(err_line, p.danger()),
                        ]));
                    }

                    // Compact colored diff for Edit tool
                    if let Some((old_s, new_s)) = edit_diff {
                        let max_diff_lines = if self.expand_tools { 40 } else { 6 };
                        render_diff_preview(old_s, new_s, max_diff_lines, content_w, p, &mut lines);
                    } else if self.expand_tools && !output_preview.is_empty() && name != "AskUser" {
                        let max_prev = 14;
                        let preview_lines: Vec<&str> = output_preview.lines().collect();
                        for pl in preview_lines.iter().take(max_prev) {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", p.border()),
                                Span::styled(
                                    truncate_display(pl, content_w.saturating_sub(8)),
                                    p.muted(),
                                ),
                            ]));
                        }
                        if preview_lines.len() > max_prev {
                            lines.push(Line::from(vec![
                                Span::styled("    │ ", p.border()),
                                Span::styled(
                                    format!("… ({} more lines)", preview_lines.len() - max_prev),
                                    p.muted(),
                                ),
                            ]));
                        }
                    }
                }
                TranscriptItem::Notice { kind, title, body } => {
                    let (icon, style) = match kind {
                        NoticeKind::Info => ("●", p.accent()),
                        NoticeKind::Success => ("✓", p.success()),
                        NoticeKind::Warning => ("▲", p.warning()),
                        NoticeKind::Error => ("✗", p.danger()),
                    };
                    lines.push(Line::from(vec![
                        Span::raw("  "),
                        Span::styled(icon, style),
                        Span::raw(" "),
                        Span::styled(title.clone(), style.add_modifier(Modifier::BOLD)),
                    ]));
                    if let Some(b) = body {
                        for bl in b.lines() {
                            lines.push(Line::from(vec![
                                Span::raw("  "),
                                Span::styled(
                                    truncate_display(bl, content_w.saturating_sub(4)),
                                    p.muted(),
                                ),
                            ]));
                        }
                    }
                }
            }
        }

        self.cached_lines = lines;
        self.cache_width = width;
        self.cache_dirty = false;
    }

    pub(crate) fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        if area.width < 20 || area.height < 8 {
            let msg = Paragraph::new("Terminal too small (min 20x8)").style(self.palette.warning());
            frame.render_widget(msg, area);
            return;
        }

        // Compute dynamic overlay & composer heights
        let slash_matches = self.slash_matches();
        let overlay_h: u16 = if self.active_perm.is_some() {
            5
        } else if let Some(ref pq) = self.active_question {
            (pq.total_rows() as u16 + 4).min(12)
        } else if let Some(ref mp) = self.model_picker {
            (mp.filtered().len().clamp(1, 8) as u16 + 4).min(13)
        } else if !slash_matches.is_empty() {
            (slash_matches.len().min(7) as u16) + 2
        } else {
            0
        };

        let composer_inner_w = area.width.saturating_sub(4) as usize;
        let max_composer_rows = 6usize;
        let preview_render = self.editor.render(composer_inner_w, max_composer_rows);
        let composer_h = (preview_render.total_rows.clamp(1, max_composer_rows) as u16) + 2;

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),          // Header
                Constraint::Min(4),             // Transcript
                Constraint::Length(overlay_h),  // Optional overlay
                Constraint::Length(composer_h), // Input box
                Constraint::Length(1),          // Status bar
            ])
            .split(area);

        self.draw_header(frame, chunks[0]);
        self.draw_transcript(frame, chunks[1]);
        if overlay_h > 0 {
            self.draw_overlay(frame, chunks[2], &slash_matches);
        }
        self.draw_composer(frame, chunks[3], &preview_render);
        self.draw_status_bar(frame, chunks[4]);
    }

    pub(crate) fn draw_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let sb_style = if self.sandbox_badge == "sandbox:off" {
            p.danger()
        } else if self.sandbox_badge == "sandbox:net" {
            p.warning()
        } else {
            p.success()
        };

        let mut left = vec![
            Span::styled(" ◆ carnot ", p.brand_bold()),
            Span::styled(format!("v{}", env!("CARGO_PKG_VERSION")), p.muted()),
            Span::styled(" · ", p.border()),
            Span::styled(self.model.clone(), p.accent()),
            Span::styled(" · ", p.border()),
            Span::styled(format!("{:?}", self.effort).to_lowercase(), p.muted()),
            Span::styled(" · ", p.border()),
            Span::styled(self.sandbox_badge, sb_style),
        ];

        let mut right_parts = Vec::new();
        if let Some(ref br) = self.git_branch {
            right_parts.push(format!("⎇ {br}"));
        }
        right_parts.push(self.workspace_short.clone());
        right_parts.push(self.session_id.chars().take(8).collect());
        let right_str = format!("{} ", right_parts.join(" · "));

        let left_w: usize = left.iter().map(|s| display_width(&s.content)).sum();
        let right_w = display_width(&right_str);
        let total_w = area.width as usize;

        if left_w + right_w + 2 <= total_w {
            left.push(Span::raw(" ".repeat(total_w - left_w - right_w)));
            left.push(Span::styled(right_str, p.muted()));
        } else if left_w + 1 < total_w {
            let rem = total_w - left_w - 1;
            if rem > 6 {
                left.push(Span::raw(" "));
                left.push(Span::styled(truncate_display(&right_str, rem), p.muted()));
            }
        }

        frame.render_widget(Paragraph::new(Line::from(left)).style(p.bar_bg()), area);
    }

    pub(crate) fn draw_transcript(&mut self, frame: &mut Frame<'_>, area: Rect) {
        self.last_viewport_height = area.height as usize;

        if self.items.is_empty() {
            self.draw_welcome(frame, area);
            return;
        }

        let width = area.width as usize;
        let has_active_spinner = self.items.iter().any(|it| {
            matches!(
                it,
                TranscriptItem::Tool {
                    status: ToolStatus::Running,
                    ..
                } | TranscriptItem::Thinking {
                    duration_ms: None,
                    ..
                }
            )
        });
        if self.cache_dirty || self.cache_width != width || has_active_spinner {
            self.rebuild_transcript_cache(width);
        }

        let total = self.cached_lines.len();
        let view_h = area.height as usize;
        let max_scroll = total.saturating_sub(view_h);
        if self.auto_follow || self.scroll_offset > max_scroll {
            self.scroll_offset = max_scroll;
        }

        let end = (self.scroll_offset + view_h).min(total);
        let visible = self.cached_lines[self.scroll_offset..end].to_vec();
        frame.render_widget(Paragraph::new(visible), area);

        if total > view_h {
            let mut sb_state = ScrollbarState::new(max_scroll).position(self.scroll_offset);
            let sb = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(None)
                .end_symbol(None)
                .track_symbol(Some("│"))
                .thumb_symbol("┃")
                .track_style(self.palette.border())
                .thumb_style(self.palette.muted());
            frame.render_stateful_widget(sb, area, &mut sb_state);
        }

        // Floating "jump to bottom" indicator when user scrolled up
        if !self.auto_follow && max_scroll > self.scroll_offset {
            let pill = " ↓ latest (End) ";
            let pw = display_width(pill) as u16;
            if area.width > pw + 2 && area.height >= 2 {
                let pill_area = Rect {
                    x: area.x + area.width - pw - 2,
                    y: area.y + area.height - 1,
                    width: pw,
                    height: 1,
                };
                frame.render_widget(Clear, pill_area);
                frame.render_widget(
                    Paragraph::new(Span::styled(pill, self.palette.selected_row())),
                    pill_area,
                );
            }
        }
    }

    pub(crate) fn draw_welcome(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let ext_line = if self.launch.active_extensions.is_empty() {
            "core only".to_string()
        } else {
            self.launch.active_extensions.join(" · ")
        };

        let mut lines = vec![
            Line::default(),
            Line::from(vec![
                Span::styled("  ◆ ", p.brand_bold()),
                Span::styled("Carnot", p.brand_bold()),
                Span::styled(
                    format!(" v{}  —  Coding Agent Harness", env!("CARGO_PKG_VERSION")),
                    p.muted(),
                ),
            ]),
            Line::default(),
            Line::from(vec![
                Span::styled("    Model       ", p.muted()),
                Span::styled(self.model.clone(), p.accent()),
                Span::styled(format!("  (effort: {:?})", self.effort), p.muted()),
            ]),
            Line::from(vec![
                Span::styled("    Workspace   ", p.muted()),
                Span::styled(self.workspace_short.clone(), p.text()),
                Span::styled(
                    self.git_branch
                        .as_deref()
                        .map(|b| format!("  ⎇ {b}"))
                        .unwrap_or_default(),
                    p.success(),
                ),
            ]),
            Line::from(vec![
                Span::styled("    Extensions  ", p.muted()),
                Span::styled(ext_line, p.text()),
            ]),
            Line::default(),
            Line::from(vec![
                Span::styled("    Tips  ", p.brand_bold()),
                Span::styled("/model", p.accent()),
                Span::styled(" or ", p.muted()),
                Span::styled("^P", p.accent()),
                Span::styled(" switch model · ", p.muted()),
                Span::styled("/approve", p.accent()),
                Span::styled(" guard tools · ", p.muted()),
                Span::styled("^T", p.accent()),
                Span::styled(" thinking · ", p.muted()),
                Span::styled("^O", p.accent()),
                Span::styled(" expand diffs", p.muted()),
            ]),
        ];

        // Vertically center lightly if plenty of space
        let top_pad = area.height.saturating_sub(lines.len() as u16) / 3;
        if top_pad > 0 {
            let mut padded = vec![Line::default(); top_pad as usize];
            padded.append(&mut lines);
            lines = padded;
        }
        frame.render_widget(Paragraph::new(lines), area);
    }

    pub(crate) fn draw_overlay(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        slash_matches: &[&SlashSpec],
    ) {
        let p = &self.palette;
        frame.render_widget(Clear, area);

        if let Some(ref perm) = self.active_perm {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.warning())
                .title(Span::styled(
                    format!(" ▲ Approve tool: {} ", perm.effective_tool),
                    p.warning().add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);
            let lines = vec![
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled(
                        truncate_display(&perm.summary, inner.width.saturating_sub(2) as usize),
                        p.text(),
                    ),
                ]),
                Line::default(),
                Line::from(vec![
                    Span::raw(" "),
                    Span::styled("[y]", p.success().add_modifier(Modifier::BOLD)),
                    Span::styled(" Allow once   ", p.muted()),
                    Span::styled("[a]", p.accent_bold()),
                    Span::styled(" Allow always   ", p.muted()),
                    Span::styled("[n / Esc]", p.danger().add_modifier(Modifier::BOLD)),
                    Span::styled(" Deny", p.muted()),
                ]),
            ];
            frame.render_widget(Paragraph::new(lines), inner);
            return;
        }

        if let Some(ref pq) = self.active_question {
            let header_tag = pq
                .current()
                .and_then(|q| q.header.as_deref())
                .map(|h| format!(" · {h}"))
                .unwrap_or_default();
            let title = format!(
                " ? Question {}/{}{header_tag} ",
                pq.current_idx + 1,
                pq.questions.len()
            );
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.warning())
                .title(Span::styled(
                    title,
                    p.warning().add_modifier(Modifier::BOLD),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            if let Some(q) = pq.current() {
                let mut lines = vec![Line::from(vec![
                    Span::raw(" "),
                    Span::styled(
                        truncate_display(&q.question, inner.width.saturating_sub(2) as usize),
                        p.text().add_modifier(Modifier::BOLD),
                    ),
                ])];
                for (idx, opt) in q.options.iter().enumerate() {
                    let is_sel = idx == pq.selected_row;
                    let marker = if q.multi_select {
                        if pq.checked.contains(&idx) {
                            "[x]"
                        } else {
                            "[ ]"
                        }
                    } else if is_sel {
                        "●"
                    } else {
                        "○"
                    };
                    let desc = opt
                        .description
                        .as_deref()
                        .map(|d| format!(" — {d}"))
                        .unwrap_or_default();
                    let row_text = format!(
                        " {} {marker} {}{desc}",
                        if is_sel { "▸" } else { " " },
                        opt.label
                    );
                    lines.push(Line::from(Span::styled(
                        truncate_display(&row_text, inner.width as usize),
                        if is_sel { p.selected_row() } else { p.text() },
                    )));
                }
                // Custom answer row
                let custom_sel = pq.is_custom_row();
                let custom_label = if pq.custom_input.is_empty() {
                    if custom_sel {
                        " ▸ Custom answer: ▏".to_string()
                    } else {
                        "   Custom answer…".to_string()
                    }
                } else {
                    format!(" ▸ Custom: {}▏", pq.custom_input)
                };
                lines.push(Line::from(Span::styled(
                    truncate_display(&custom_label, inner.width as usize),
                    if custom_sel {
                        p.selected_row()
                    } else {
                        p.muted()
                    },
                )));
                lines.push(Line::from(Span::styled(
                    if q.multi_select {
                        " ↑↓ move · Space toggle · Enter submit · Esc skip"
                    } else {
                        " ↑↓ move · 1-9 pick · Enter submit · Esc skip"
                    },
                    p.muted(),
                )));
                frame.render_widget(Paragraph::new(lines), inner);
            }
            return;
        }

        if let Some(ref mp) = self.model_picker {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.accent())
                .title(Span::styled(
                    " ◆ Select Model (type to filter · Enter switch · Esc close) ",
                    p.accent_bold(),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let mut lines = vec![Line::from(vec![
                Span::styled(" Filter: ", p.muted()),
                Span::styled(
                    if mp.filter.is_empty() {
                        "type model or provider…".to_string()
                    } else {
                        format!("{}▏", mp.filter)
                    },
                    if mp.filter.is_empty() {
                        p.muted()
                    } else {
                        p.text()
                    },
                ),
            ])];

            let filtered = mp.filtered();
            if filtered.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  (no matching models)",
                    p.muted(),
                )));
            } else {
                let max_show = inner.height.saturating_sub(1) as usize;
                let sel = mp.selected.min(filtered.len() - 1);
                let start = sel.saturating_sub(max_show.saturating_sub(1));
                for (offset, m) in filtered.iter().skip(start).take(max_show).enumerate() {
                    let idx = start + offset;
                    let is_sel = idx == sel;
                    let active_dot = if m.id == self.model { "●" } else { " " };
                    let text = format!(" {active_dot} {:<28}  [{}]", m.id, m.provider);
                    lines.push(Line::from(Span::styled(
                        truncate_display(&text, inner.width as usize),
                        if is_sel {
                            p.selected_row()
                        } else if m.id == self.model {
                            p.success()
                        } else {
                            p.text()
                        },
                    )));
                }
            }
            frame.render_widget(Paragraph::new(lines), inner);
            return;
        }

        if !slash_matches.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(p.border())
                .title(Span::styled(
                    " Commands (Tab complete · Enter run) ",
                    p.muted(),
                ));
            let inner = block.inner(area);
            frame.render_widget(block, area);

            let sel = self.slash_selected.min(slash_matches.len() - 1);
            let lines: Vec<Line<'static>> = slash_matches
                .iter()
                .take(7)
                .enumerate()
                .map(|(idx, s)| {
                    let row = format!(" {:<18} {}", s.usage, s.desc);
                    Line::from(Span::styled(
                        truncate_display(&row, inner.width as usize),
                        if idx == sel {
                            p.selected_row()
                        } else {
                            p.text()
                        },
                    ))
                })
                .collect();
            frame.render_widget(Paragraph::new(lines), inner);
        }
    }

    pub(crate) fn draw_composer(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        rendered: &editor::RenderedEditor,
    ) {
        let p = &self.palette;
        let has_modal = self.active_perm.is_some()
            || self.active_question.is_some()
            || self.model_picker.is_some();

        let (border_style, title_span) = if self.running {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            let secs = self
                .turn_started_at
                .map(|t| format!("{:.1}s", t.elapsed().as_secs_f64()))
                .unwrap_or_default();
            (
                p.warning(),
                Span::styled(
                    format!(" {spin} Working… ({secs}) · Esc to abort "),
                    p.warning().add_modifier(Modifier::BOLD),
                ),
            )
        } else if self.compacting {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            (
                p.warning(),
                Span::styled(format!(" {spin} Compacting context… "), p.warning()),
            )
        } else if has_modal {
            (p.border(), Span::styled(" › Prompt ", p.muted()))
        } else {
            (p.brand_bold(), Span::styled(" › Prompt ", p.brand_bold()))
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(border_style)
            .title(title_span);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        if self.editor.is_empty() {
            let placeholder = if self.running {
                "Agent is running… press Esc or Ctrl+C to abort turn"
            } else {
                "Ask anything… (/ commands · Shift+Enter newline · ^P model)"
            };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    truncate_display(placeholder, inner.width as usize),
                    p.muted(),
                )),
                inner,
            );
        } else {
            let lines: Vec<Line<'static>> = rendered
                .lines
                .iter()
                .map(|s| Line::from(Span::styled(s.clone(), p.text())))
                .collect();
            frame.render_widget(Paragraph::new(lines), inner);
        }

        if !has_modal && inner.width > 0 && inner.height > 0 {
            let cx = inner.x + rendered.cursor_col.min(inner.width.saturating_sub(1));
            let cy = inner.y + rendered.cursor_row.min(inner.height.saturating_sub(1));
            frame.set_cursor_position((cx, cy));
        }
    }

    pub(crate) fn draw_status_bar(&self, frame: &mut Frame<'_>, area: Rect) {
        let p = &self.palette;
        let mut left = Vec::new();

        if let Some(prev) = self.last_ctrl_c
            && prev.elapsed() < Duration::from_millis(1600)
            && !self.running
        {
            left.push(Span::styled(" Press Ctrl+C again to quit ", p.warning()));
        } else if self.running {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            left.push(Span::styled(format!(" {spin} running "), p.warning()));
        } else if self.compacting {
            let spin = SPINNER[self.spinner_frame % SPINNER.len()];
            left.push(Span::styled(format!(" {spin} compacting "), p.warning()));
        } else {
            left.push(Span::styled(" ● ready ", p.success()));
        }

        if self.approve_enabled {
            left.push(Span::styled(" approve:on ", p.warning()));
        }
        if self.show_thinking {
            left.push(Span::styled(" think:on ", p.thinking()));
        }
        if self.expand_tools {
            left.push(Span::styled(" expand:on ", p.accent()));
        }
        if !self.workers.is_empty() {
            let w_run = self
                .workers
                .values()
                .filter(|w| w.state == "running")
                .count();
            let w_idle = self.workers.values().filter(|w| w.state == "idle").count();
            let chip = if w_run > 0 && w_idle > 0 {
                format!(" fleet:{w_run}⚡/{w_idle}⏸ ")
            } else if w_run > 0 {
                format!(" fleet:{w_run}⚡ ")
            } else {
                format!(" fleet:{w_idle}⏸ ")
            };
            let style = if w_run > 0 { p.warning() } else { p.accent() };
            left.push(Span::styled(chip, style));
        }
        if self.cache_breaks > 0 {
            left.push(Span::styled(
                format!(" cache-breaks:{} ", self.cache_breaks),
                p.danger(),
            ));
        }
        let ext_span = if self.launch.active_extensions.is_empty() {
            None
        } else {
            let ext_chip = self
                .launch
                .active_extensions
                .iter()
                .map(|e| format!("+{e}"))
                .collect::<Vec<_>>()
                .join(" ");
            Some(Span::styled(format!(" {ext_chip} "), p.muted()))
        };

        // Context meter + cache + cost on the right
        let pct = ((self.ctx_tokens as f64 / self.ctx_window.max(1) as f64) * 100.0).min(100.0);
        let filled = ((pct / 100.0) * 6.0).round() as usize;
        let meter = format!(
            "{}{}",
            "█".repeat(filled.min(6)),
            "░".repeat(6usize.saturating_sub(filled))
        );
        let ctx_style = if pct >= 85.0 {
            p.danger()
        } else if pct >= 60.0 {
            p.warning()
        } else {
            p.success()
        };

        let right_spans = vec![
            Span::styled(
                format!(
                    "ctx {}/{} ",
                    fmt_tokens(self.ctx_tokens),
                    fmt_tokens(self.ctx_window)
                ),
                p.muted(),
            ),
            Span::styled(format!("[{meter}] {pct:.0}%"), ctx_style),
            Span::styled(
                format!(
                    " · cache {:.0}% · ${:.4} ",
                    self.cache_hit_pct, self.cost_usd
                ),
                p.muted(),
            ),
        ];

        let total_w = area.width as usize;
        let right_w: usize = right_spans.iter().map(|s| display_width(&s.content)).sum();

        // Include the `+ext` chip only when it fits comfortably alongside left & right status
        if let Some(ext) = ext_span {
            let lw: usize = left.iter().map(|s| display_width(&s.content)).sum();
            let ew = display_width(&ext.content);
            if lw + ew + right_w < total_w {
                left.push(ext);
            }
        }

        let left_w: usize = left.iter().map(|s| display_width(&s.content)).sum();
        if left_w + right_w < total_w {
            left.push(Span::raw(" ".repeat(total_w - left_w - right_w)));
            left.extend(right_spans);
        } else if left_w + 18 < total_w {
            // Compact right fallback for narrow split panes (~60-80 cols)
            let compact_right = vec![
                Span::styled(format!("{pct:.0}% "), ctx_style),
                Span::styled(format!("${:.4} ", self.cost_usd), p.muted()),
            ];
            let cw: usize = compact_right
                .iter()
                .map(|s| display_width(&s.content))
                .sum();
            if left_w + cw < total_w {
                left.push(Span::raw(" ".repeat(total_w - left_w - cw)));
                left.extend(compact_right);
            }
        } else if right_w < total_w {
            left = vec![Span::raw(" ".repeat(total_w - right_w))];
            left.extend(right_spans);
        }

        frame.render_widget(Paragraph::new(Line::from(left)).style(p.bar_bg()), area);
    }
}
