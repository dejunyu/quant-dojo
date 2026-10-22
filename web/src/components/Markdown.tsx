import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

/**
 * Markdown 渲染器：支持 GFM（表格、删除线）与 KaTeX 数学公式。
 * 课程里的 `$...$` 与 `$$...$$` 都会在这里渲染成公式。
 */
export function Markdown({ children }: { children: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { strict: false, throwOnError: false }]]}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
