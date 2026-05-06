// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightThemeRapide from 'starlight-theme-rapide';

const base = process.env.BASE_PATH ?? '/';
const site = process.env.SITE_URL ?? 'https://jd-opensource.github.io/xllm';

// https://astro.build/config
export default defineConfig({
	site,
	base,
	integrations: [
		starlight({
			plugins: [starlightThemeRapide()],
			customCss: ['./src/styles/theme.css'],
			components: {
				Header: './src/components/Header.astro',
				PageTitle: './src/components/PageTitle.astro',
			},
			title: {
				en: 'xLLM',
				'zh-CN': 'xLLM',
			},
			locales: {
				en: {
					label: 'EN',
					lang: 'en',
				},
				zh: {
					label: '中',
					lang: 'zh-CN',
				},
			},
			defaultLocale: 'en',
			logo: {
				src: './src/assets/logo_with_llm.png',
				alt: 'xLLM',
				replacesTitle: true,
			},
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/jd-opensource/xllm' }],
			sidebar: [
				{ label: 'Home', translations: { 'zh-CN': '主页' }, slug: '' },
				{
					label: 'User Guide',
					translations: { 'zh-CN': '用户指南' },
					items: [
						{
							label: 'Getting Started',
							translations: { 'zh-CN': '开始使用' },
							autogenerate: { directory: 'getting_started' },
						},
						{
							label: 'Supported Models',
							translations: { 'zh-CN': '模型支持列表' },
							slug: 'supported_models',
						},
						{
							label: 'Popular Model Usage',
							translations: { 'zh-CN': '热门模型使用' },
							autogenerate: { directory: 'popular_model_usage' },
						},
					],
				},
				{
					label: 'Advanced Features',
					translations: { 'zh-CN': '高级功能' },
					autogenerate: { directory: 'features' },
				},
				{
					label: 'Developer Guide',
					translations: { 'zh-CN': '开发者指南' },
					items: [
						{
							label: 'Development',
							translations: { 'zh-CN': '开发' },
							autogenerate: { directory: 'dev_guide' },
						},
						{
							label: 'Design',
							translations: { 'zh-CN': '设计文档' },
							autogenerate: { directory: 'design' },
						},
					],
				},
				{ label: 'CLI Reference', translations: { 'zh-CN': 'CLI 参考' }, slug: 'cli_reference' },
			],
		}),
	],
});
