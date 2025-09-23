// components/MdxComponents.tsx
import React from 'react'

export const MdxComponents = {
  h1: (p: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h1 {...p} className="text-2xl font-bold mt-6 mb-3" />
  ),
  h2: (p: React.HTMLAttributes<HTMLHeadingElement>) => (
    <h2 {...p} className="text-xl font-semibold mt-5 mb-2" />
  ),
  p: (p: React.HTMLAttributes<HTMLParagraphElement>) => (
    <p {...p} className="leading-7 my-3" />
  ),
  code: (p: React.HTMLAttributes<HTMLElement>) => (
    <code {...p} className="rounded bg-gray-100 px-1 py-0.5 text-sm" />
  ),
  pre: (p: React.HTMLAttributes<HTMLPreElement>) => (
    <pre {...p} className="rounded-xl bg-gray-900 text-gray-100 p-4 overflow-x-auto my-4" />
  ),
  ul: (p: React.HTMLAttributes<HTMLUListElement>) => (
    <ul {...p} className="list-disc pl-6 my-3" />
  ),
  ol: (p: React.HTMLAttributes<HTMLOListElement>) => (
    <ol {...p} className="list-decimal pl-6 my-3" />
  ),
  a: (p: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a {...p} className="text-blue-600 underline" />
  ),
}
