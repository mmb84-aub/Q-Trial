import type { SynthesisOutput } from "../../types";

interface Props {
  synthesis: SynthesisOutput;
}

export function ForwardRecommendations({ synthesis }: Props) {
  return (
    <section aria-labelledby="forward-title" style={{ marginBottom: "2rem" }}>
      <h2 id="forward-title">Forward Recommendations</h2>
      <p><strong>Future Trial Hypothesis:</strong> {synthesis.future_trial_hypothesis}</p>
      <p><strong>Recommended Sample Size:</strong> {synthesis.recommended_sample_size}</p>
      {synthesis.endpoint_improvement_recommendations.length > 0 && (
        <>
          <h3>Endpoint Improvements</h3>
          <ul>
            {synthesis.endpoint_improvement_recommendations.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </>
      )}
      {synthesis.variables_to_control.length > 0 && (
        <>
          <h3>Variables to Control</h3>
          <ul>
            {synthesis.variables_to_control.map((v, i) => (
              <li key={i}><strong>{v.variable}</strong>: {v.reason}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
