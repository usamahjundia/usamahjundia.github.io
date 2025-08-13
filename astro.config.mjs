// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, passthroughImageService } from 'astro/config';

// https://astro.build/config
export default defineConfig({
	site: 'https://usamahjundia.github.io',
	output: 'static',
	vite: {
		plugins: [tailwindcss()],
	},
	image: {
	service: passthroughImageService(),
	},
});
