import { useState } from "react";
import { Check, Copy, ExternalLink } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Citation({ citation }) {
  const [copied, setCopied] = useState(false);
  const bibtex = citation.bibtex || citation.text;
  const copy = async () => {
    await navigator.clipboard?.writeText(bibtex).catch(() => undefined);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <section className="section" id="cite">
      <div className="container">
        <div className="section-head"><div className="section-tag">Citation</div><h2>Cite the paper</h2></div>
        <div className="cite-block">
          <div className="cite-reference">
            <span className="cite-label">Recommended reference</span>
            <h3>{citation.title}</h3>
            {citation.authors && <p className="cite-authors">{citation.authors}</p>}
            <div className="cite-meta">
              {citation.venue && <span>{citation.venue}</span>}
              {citation.arxiv && <span>arXiv:{citation.arxiv}</span>}
              {citation.year && <span>{citation.year}</span>}
            </div>
            {citation.reference && <p className="cite-readable">{citation.reference}</p>}
            <div className="cite-actions">
              <Button type="button" variant="brand" onClick={copy}>{copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}{copied ? "Copied" : "Copy BibTeX"}</Button>
              {citation.paperUrl && <Button asChild variant="ghost"><a href={citation.paperUrl} target="_blank" rel="noreferrer">Read paper <ExternalLink size={15} aria-hidden="true" /></a></Button>}
            </div>
          </div>
          <div className="cite-code-panel" aria-label="BibTeX citation">
            <div className="cite-code-head"><span>BibTeX</span><button className="copy-btn" type="button" onClick={copy}>{copied ? "Copied" : "Copy"}</button></div>
            <pre>{bibtex}</pre>
          </div>
        </div>
      </div>
    </section>
  );
}