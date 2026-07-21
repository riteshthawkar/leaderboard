import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { FileDropzone } from "@/components/FileDropzone";


describe("FileDropzone", () => {
  it("rejects a selected file above the configured in-memory limit", () => {
    render(
      <FileDropzone
        name="file"
        accept=".zip"
        maxBytes={3}
        hint="Spatial package"
      />,
    );
    const input = document.querySelector('input[name="file"]');

    fireEvent.change(input, {
      target: {
        files: [new File(["1234"], "spatial_reasoning_submission.zip", { type: "application/zip" })],
      },
    });

    expect(screen.getByRole("alert")).toHaveTextContent(
      "File too large. Maximum size is 3 B.",
    );
    expect(screen.queryByText(/ready to submit/)).not.toBeInTheDocument();
  });
});
