import { useRef, useState } from "react";
import { UploadCloud, FileText, X } from "lucide-react";

function formatBytes(bytes) {
  if (bytes == null) return "";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(1)} ${units[i]}`;
}

export function FileDropzone({ name, accept = "", required = false, hint }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [error, setError] = useState("");

  const acceptList = accept
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean);

  const isAllowed = (candidate) =>
    !acceptList.length ||
    acceptList.some((ext) => candidate.name.toLowerCase().endsWith(ext));

  const syncFromInput = () => {
    const picked = inputRef.current?.files?.[0];
    if (!picked) {
      setFile(null);
      return;
    }
    if (!isAllowed(picked)) {
      setError(`Unsupported file. Use ${acceptList.join(" or ")}.`);
      if (inputRef.current) inputRef.current.value = "";
      setFile(null);
      return;
    }
    setError("");
    setFile({ name: picked.name, size: picked.size });
  };

  const clear = () => {
    if (inputRef.current) inputRef.current.value = "";
    setFile(null);
    setError("");
  };

  return (
    <div className="dropzone-wrap">
      <div
        className={`dropzone${dragOver ? " is-dragover" : ""}${file ? " has-file" : ""}${error ? " has-error" : ""}`}
      >
        <input
          ref={inputRef}
          type="file"
          name={name}
          accept={accept}
          required={required}
          className="dropzone-input"
          onChange={syncFromInput}
          onDragEnter={() => setDragOver(true)}
          onDragOver={() => setDragOver(true)}
          onDragLeave={() => setDragOver(false)}
          onDrop={() => setDragOver(false)}
        />
        {file ? (
          <div className="dropzone-file">
            <FileText className="dropzone-file-icon" size={22} aria-hidden="true" />
            <span className="dropzone-file-meta">
              <span className="dropzone-file-name">{file.name}</span>
              <span className="dropzone-file-size">
                {formatBytes(file.size)} · ready to submit
              </span>
            </span>
            <button
              type="button"
              className="dropzone-remove"
              onClick={clear}
              aria-label="Remove selected file"
            >
              <X size={16} aria-hidden="true" />
            </button>
          </div>
        ) : (
          <div className="dropzone-prompt">
            <span className="dropzone-icon">
              <UploadCloud size={24} aria-hidden="true" />
            </span>
            <span className="dropzone-text">
              <strong>Click to upload</strong> or drag and drop
            </span>
            {hint && <span className="dropzone-hint">{hint}</span>}
          </div>
        )}
      </div>
      {error && (
        <p className="dropzone-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
