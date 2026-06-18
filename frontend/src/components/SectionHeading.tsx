import './SectionHeading.css';

/* ──── SectionHeading ──── */

export interface SectionHeadingProps {
  text: string;
  level: number;
  collapsed: boolean;
}

/** 标题行组件。
 *
 * 各级标题使用独立的 CSS class ``mutagent-section-heading--level1`` ~ ``--level6``，
 * 可通过覆盖对应 class 单独定制字号、字重、间距、缩进。
 */
export function SectionHeading({ text, level }: SectionHeadingProps) {
  return (
    <div className={`mutagent-section-heading mutagent-section-heading--level${level}`}>
      <span className="mutagent-section-heading__text">{text}</span>
    </div>
  );
}
