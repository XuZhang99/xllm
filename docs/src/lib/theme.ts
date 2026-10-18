// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

/** Theme preference stored for the docs site. `auto` follows the operating system. */
export type ThemePreference = 'auto' | 'dark' | 'light';

/** Concrete theme applied to the page. */
export type ResolvedTheme = 'dark' | 'light';

/** Normalizes an arbitrary stored value into a supported theme preference. */
export function parseThemePreference(value: unknown): ThemePreference {
	return value === 'dark' || value === 'light' ? value : 'auto';
}

/** Resolves a preference against the current system appearance. */
export function resolveTheme(preference: ThemePreference, systemTheme: ResolvedTheme): ResolvedTheme {
	return preference === 'auto' ? systemTheme : preference;
}

/** Returns the concrete theme a toggle click should apply. */
export function getNextTheme(preference: ThemePreference, systemTheme: ResolvedTheme): ResolvedTheme {
	return resolveTheme(preference, systemTheme) === 'dark' ? 'light' : 'dark';
}

/**
 * Returns the preference to persist after the user selects `selected`.
 *
 * Picking the theme the system already uses switches back to `auto`, so the site
 * resumes following later system appearance changes. This is the only way out of
 * an explicitly pinned theme and mirrors the main xLLM website.
 */
export function getPreferenceToStore(selected: ResolvedTheme, systemTheme: ResolvedTheme): ThemePreference {
	return selected === systemTheme ? 'auto' : selected;
}
