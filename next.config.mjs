import path from 'path'
import createMDX from '@next/mdx'
import remarkGfm from 'remark-gfm'
import remarkFrontmatter from 'remark-frontmatter'
import remarkMdxFrontmatter from 'remark-mdx-frontmatter'

const withMDX = createMDX({
  extension: /\.mdx?$/,
  options: {
    remarkPlugins: [remarkGfm, remarkFrontmatter, remarkMdxFrontmatter],
    rehypePlugins: [],
  },
})

/** @type {import('next').NextConfig} */
const baseConfig = {
  pageExtensions: ['js', 'jsx', 'ts', 'tsx', 'md', 'mdx'],

  // ← これで「workspace root を誤推定した」警告を抑止
  outputFileTracingRoot: path.resolve(process.cwd(), '../../'),

  // （必要なら）CIで ESLint をビルドブロッカーにしない
  // eslint: { ignoreDuringBuilds: true },
}

export default withMDX(baseConfig)