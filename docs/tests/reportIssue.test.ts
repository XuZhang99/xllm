// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { buildReportIssueUrl, getReportIssueLabel } from '../src/lib/reportIssue.ts';

test('the button label follows the page locale', () => {
	assert.equal(getReportIssueLabel('en'), 'Report issue');
	assert.equal(getReportIssueLabel('zh'), '反馈文档问题');
	assert.equal(getReportIssueLabel('zh-CN'), '反馈文档问题');
	assert.equal(getReportIssueLabel(undefined), 'Report issue');
});

test('the issue title carries the page path quoted like the vLLM Ascend docs', () => {
	const link = buildReportIssueUrl('/en/dev_guide/tilelang_ascend_kernel_dev/');
	const url = new URL(link);

	assert.equal(
		url.searchParams.get('title'),
		'[Docs]: Feedback for `/en/dev_guide/tilelang_ascend_kernel_dev/`',
	);
	// Paths without a leading slash are normalized so the quoted path is absolute.
	assert.equal(
		new URL(buildReportIssueUrl('zh/features/disagg_pd/')).searchParams.get('title'),
		'[Docs]: Feedback for `/zh/features/disagg_pd/`',
	);
});

test('the report link opens the documentation issue form', () => {
	const link = buildReportIssueUrl('/en/getting_started/quick_start/');
	const url = new URL(link);

	assert.equal(`${url.origin}${url.pathname}`, 'https://github.com/xLLM-AI/xllm/issues/new');
	assert.equal(url.searchParams.get('template'), 'documentation.yaml');
	// The form applies the label itself, so the link only selects the template.
	assert.deepEqual([...url.searchParams.keys()].sort(), ['template', 'title']);
	// The query is encoded by hand so spaces stay `%20` instead of becoming `+`.
	assert.ok(!link.includes('+'));
});

test('the linked issue form exists and files documentation issues', () => {
	const template = new URL(buildReportIssueUrl('/en/')).searchParams.get('template');
	const source = readFileSync(
		new URL(`../../.github/ISSUE_TEMPLATE/${template}`, import.meta.url),
		'utf-8',
	);

	assert.match(source, /^name: 📚 Documentation$/m);
	assert.match(source, /^labels: \["documentation"\]$/m);
});
