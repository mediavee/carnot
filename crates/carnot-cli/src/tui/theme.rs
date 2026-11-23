//! Cohesive modern dark-terminal palette inspired by Claude Code & OpenCode,
//! with automatic `NO_COLOR` fallback.

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Palette {
    pub(crate) color: bool,
    /// Warm terracotta/amber brand accent (`◆ carnot`, active borders, headings).
    pub(crate) brand: Color,
    /// Soft sky/cyan accent (user messages, inline code, links).
    pub(crate) accent: Color,
    /// Emerald green (tool success, diff additions, active badges).
    pub(crate) success: Color,
    /// Coral red (errors, diff deletions).
    pub(crate) danger: Color,
    /// Warm gold (warnings, `AskUser`, approvals, compaction).
    pub(crate) warning: Color,
    /// Soft violet (thinking / reasoning).
    pub(crate) thinking: Color,
    /// Primary readable foreground.
    pub(crate) fg: Color,
    /// Secondary muted foreground (tool args, metadata, timestamps).
    pub(crate) muted: Color,
    /// Subtle structural borders and dividers.
    pub(crate) border: Color,
    /// Subtle surface tint for code blocks / header / user prompt gutter.
    pub(crate) surface: Color,
    /// Slightly elevated surface for popups / selected rows.
    pub(crate) surface_hi: Color,
    /// Syntax highlight colors.
    pub(crate) syn_keyword: Color,
    pub(crate) syn_string: Color,
    pub(crate) syn_number: Color,
    pub(crate) syn_comment: Color,
    pub(crate) syn_type: Color,
}

impl Palette {
    pub(crate) fn detect() -> Self {
        let no_color = std::env::var_os("NO_COLOR").is_some();
        if no_color {
            return Self {
                color: false,
                brand: Color::Reset,
                accent: Color::Reset,
                success: Color::Reset,
                danger: Color::Reset,
                warning: Color::Reset,
                thinking: Color::Reset,
                fg: Color::Reset,
                muted: Color::Reset,
                border: Color::Reset,
                surface: Color::Reset,
                surface_hi: Color::Reset,
                syn_keyword: Color::Reset,
                syn_string: Color::Reset,
                syn_number: Color::Reset,
                syn_comment: Color::Reset,
                syn_type: Color::Reset,
            };
        }

        // Modern terminals (Ghostty, iTerm2, Kitty, WezTerm, Alacritty, Warp, macOS Terminal 256/TC)
        // handle RGB smoothly; fall back gracefully only on explicit TERM=vt100/dumb.
        let basic = matches!(
            std::env::var("TERM").ok().as_deref(),
            Some("dumb" | "vt100" | "ansi")
        );
        if basic {
            return Self {
                color: true,
                brand: Color::Yellow,
                accent: Color::Cyan,
                success: Color::Green,
                danger: Color::Red,
                warning: Color::Yellow,
                thinking: Color::Magenta,
                fg: Color::Reset,
                muted: Color::DarkGray,
                border: Color::DarkGray,
                surface: Color::Reset,
                surface_hi: Color::DarkGray,
                syn_keyword: Color::Magenta,
                syn_string: Color::Green,
                syn_number: Color::Yellow,
                syn_comment: Color::DarkGray,
                syn_type: Color::Cyan,
            };
        }

        Self {
            color: true,
            brand: Color::Rgb(235, 132, 72),
            accent: Color::Rgb(115, 192, 245),
            success: Color::Rgb(116, 206, 146),
            danger: Color::Rgb(238, 114, 114),
            warning: Color::Rgb(240, 198, 116),
            thinking: Color::Rgb(176, 152, 226),
            fg: Color::Reset,
            muted: Color::Rgb(138, 144, 158),
            border: Color::Rgb(68, 73, 86),
            surface: Color::Rgb(28, 31, 38),
            surface_hi: Color::Rgb(42, 47, 58),
            syn_keyword: Color::Rgb(198, 148, 240),
            syn_string: Color::Rgb(152, 208, 146),
            syn_number: Color::Rgb(238, 178, 114),
            syn_comment: Color::Rgb(112, 120, 136),
            syn_type: Color::Rgb(126, 202, 224),
        }
    }

    pub(crate) fn text(&self) -> Style {
        Style::default().fg(self.fg)
    }

    pub(crate) fn muted(&self) -> Style {
        if self.color {
            Style::default().fg(self.muted)
        } else {
            Style::default().add_modifier(Modifier::DIM)
        }
    }

    pub(crate) fn brand_bold(&self) -> Style {
        Style::default().fg(self.brand).add_modifier(Modifier::BOLD)
    }

    pub(crate) fn accent(&self) -> Style {
        Style::default().fg(self.accent)
    }

    pub(crate) fn accent_bold(&self) -> Style {
        Style::default()
            .fg(self.accent)
            .add_modifier(Modifier::BOLD)
    }

    pub(crate) fn success(&self) -> Style {
        Style::default().fg(self.success)
    }

    pub(crate) fn danger(&self) -> Style {
        Style::default().fg(self.danger)
    }

    pub(crate) fn warning(&self) -> Style {
        Style::default().fg(self.warning)
    }

    pub(crate) fn thinking(&self) -> Style {
        Style::default()
            .fg(self.thinking)
            .add_modifier(Modifier::ITALIC)
    }

    pub(crate) fn border(&self) -> Style {
        Style::default().fg(self.border)
    }

    pub(crate) fn code_inline(&self) -> Style {
        if self.color {
            Style::default().fg(self.accent).bg(self.surface)
        } else {
            Style::default().add_modifier(Modifier::BOLD)
        }
    }

    pub(crate) fn bar_bg(&self) -> Style {
        if self.color {
            Style::default().bg(self.surface).fg(self.fg)
        } else {
            Style::default().add_modifier(Modifier::REVERSED)
        }
    }

    pub(crate) fn selected_row(&self) -> Style {
        if self.color {
            Style::default()
                .bg(self.surface_hi)
                .fg(self.brand)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD)
        }
    }
}
