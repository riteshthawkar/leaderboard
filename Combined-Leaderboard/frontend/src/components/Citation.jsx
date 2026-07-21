import { useState } from "react";
import { ArrowUpRight, Check, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ui } from "@/lib/styles";
import { cn } from "@/lib/utils";

export function Citation({ citation }) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState("");
  const bibtex = citation.bibtex || citation.text;
  const copy = async () => {
    setCopyError("");
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard access is unavailable");
      await navigator.clipboard.writeText(bibtex);
      setCopied(true);
      setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
      setCopyError("The citation could not be copied automatically. Select the BibTeX text and copy it manually.");
    }
  };

  return (
    <section className="pb-16 lg:pb-20" id="cite">
      <div className={cn(ui.sectionFrame, "!border-page-accent-border")}>
        <div className={cn(ui.sectionBand, "!border-page-accent-border")}><div className="max-w-copy"><div className="mb-3 text-xs font-semibold uppercase text-page-accent">Citation</div><h2 className={ui.heading2}>Cite the paper</h2></div></div>
        <div className="grid min-w-0 border border-border lg:grid-cols-2">
          <div className="border-b border-border p-8 lg:border-b-0 lg:border-r">
            {citation.reference && <p className="m-0 max-w-[60ch] leading-loose text-foreground">{citation.reference}</p>}
            <div className="mt-7 flex flex-wrap gap-2.5">
              <Button type="button" variant="brand" onClick={copy}>{copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}{copied ? "Copied" : "Copy BibTeX"}</Button>
              {citation.paperUrl && <Button asChild variant="ghost"><a href={citation.paperUrl} target="_blank" rel="noreferrer">Read paper <ArrowUpRight className="size-3.5" strokeWidth={1.75} aria-hidden="true" /></a></Button>}
            </div>
            {copyError && <div className="mt-3 border border-negative border-l-[3px] bg-negative-soft p-3 text-sm text-negative" role="alert">{copyError}</div>}
          </div>
          <div className="grid min-w-0 grid-rows-[auto_1fr]" aria-label="BibTeX citation">
            <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3"><span className="text-xs font-semibold uppercase text-faint">BibTeX</span><button className="border border-border bg-surface px-2.5 py-1 text-xs text-muted hover:text-foreground" type="button" onClick={copy}>{copied ? "Copied" : "Copy"}</button></div>
            <pre className="m-0 overflow-x-auto p-5 font-sans text-xs leading-relaxed text-muted">{bibtex}</pre>
          </div>
        </div>
      </div>
    </section>
  );
}
