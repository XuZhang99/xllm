// Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.

/** Repository that hosts the documentation sources and receives issue reports. */
export const DOCUMENTATION_REPOSITORY = 'xLLM-AI/xllm';

/** Issue form GitHub opens for reports coming from the documentation site. */
const ISSUE_TEMPLATE = 'documentation.yaml';

/**
 * Returns the label of the report issue button for a locale.
 *
 * Starlight locale codes are matched by prefix so that `zh` and `zh-CN` are
 * treated the same.
 */
export function getReportIssueLabel(locale: string | undefined): string {
	return locale?.startsWith('zh') ? '反馈文档问题' : 'Report issue';
}

/**
 * Builds a prefilled GitHub "new issue" URL for a documentation page.
 *
 * The page path goes into the title, quoted the same way as the vLLM Ascend
 * docs, because GitHub only prefills the title for issue forms; the body is
 * collected through the form itself. The `documentation` label is set by the
 * form, so the link only has to select the template.
 */
export function buildReportIssueUrl(pathname: string): string {
	const path = pathname.startsWith('/') ? pathname : `/${pathname}`;
	const query = [
		['template', ISSUE_TEMPLATE],
		['title', `[Docs]: Feedback for \`${path}\``],
	]
		.map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
		.join('&');

	return `https://github.com/${DOCUMENTATION_REPOSITORY}/issues/new?${query}`;
}
