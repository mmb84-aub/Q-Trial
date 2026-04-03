import type { ResearchQuestion } from "../../types";

interface Props {
  questions: ResearchQuestion[];
}

export function QuestionBank({ questions }: Props) {
  if (questions.length === 0) return null;
  return (
    <section aria-labelledby="questions-title" style={{ marginBottom: "2rem" }}>
      <h2 id="questions-title">Research Question Bank</h2>
      <ol>
        {questions.map((q, i) => (
          <li key={i} style={{ marginBottom: "0.5rem" }}>
            <strong>{q.question}</strong>
            <br />
            <span style={{ fontSize: "0.85rem", color: "#555" }}>
              Source: {q.source_finding}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
