// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
	getNextTheme,
	getPreferenceToStore,
	parseThemePreference,
	resolveTheme,
	type ResolvedTheme,
	type ThemePreference,
} from '../src/lib/theme.ts';

test('stored values are normalized to a supported preference', () => {
	assert.equal(parseThemePreference('dark'), 'dark');
	assert.equal(parseThemePreference('light'), 'light');
	assert.equal(parseThemePreference('auto'), 'auto');
	for (const value of [null, undefined, '', 'system', 'Dark', 1]) {
		assert.equal(parseThemePreference(value), 'auto');
	}
});

test('automatic mode follows the system appearance, explicit modes do not', () => {
	for (const systemTheme of ['light', 'dark'] as const) {
		assert.equal(resolveTheme('auto', systemTheme), systemTheme);
		assert.equal(resolveTheme('light', systemTheme), 'light');
		assert.equal(resolveTheme('dark', systemTheme), 'dark');
	}
});

test('the toggle flips the currently resolved appearance', () => {
	assert.equal(getNextTheme('auto', 'light'), 'dark');
	assert.equal(getNextTheme('auto', 'dark'), 'light');
	assert.equal(getNextTheme('light', 'light'), 'dark');
	assert.equal(getNextTheme('light', 'dark'), 'dark');
	assert.equal(getNextTheme('dark', 'light'), 'light');
	assert.equal(getNextTheme('dark', 'dark'), 'light');
});

test('selecting the theme the system already uses resumes automatic mode', () => {
	assert.equal(getPreferenceToStore('light', 'light'), 'auto');
	assert.equal(getPreferenceToStore('dark', 'dark'), 'auto');
	assert.equal(getPreferenceToStore('light', 'dark'), 'light');
	assert.equal(getPreferenceToStore('dark', 'light'), 'dark');
});

test('an explicitly pinned theme can always return to automatic mode', () => {
	/** Simulates the page state after each click: stored preference and applied theme. */
	function click(preference: ThemePreference, systemTheme: ResolvedTheme) {
		const selected = getNextTheme(preference, systemTheme);
		const nextPreference = getPreferenceToStore(selected, systemTheme);
		return { preference: nextPreference, theme: resolveTheme(nextPreference, systemTheme) };
	}

	for (const systemTheme of ['light', 'dark'] as const) {
		// Starting from automatic mode, two clicks always come back to automatic mode.
		let state = click('auto', systemTheme);
		assert.notEqual(state.preference, 'auto');
		state = click(state.preference, systemTheme);
		assert.equal(state.preference, 'auto');
		assert.equal(state.theme, systemTheme);

		// A theme pinned to the opposite of the system returns in one click.
		const opposite = systemTheme === 'light' ? 'dark' : 'light';
		state = click(opposite, systemTheme);
		assert.equal(state.preference, 'auto');
		assert.equal(state.theme, systemTheme);

		// A theme pinned to the system value also returns in two clicks.
		state = click(systemTheme, systemTheme);
		assert.equal(state.preference, opposite);
		state = click(state.preference, systemTheme);
		assert.equal(state.preference, 'auto');
	}
});
