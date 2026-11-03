import { useCallback, useState, type DragEvent } from "react";
import styles from "./Dropzone.module.css";

export function Dropzone({
  onFile,
  accept = "image/*",
}: {
  onFile: (file: File) => void;
  accept?: string;
}) {
  const [dragOver, setDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLLabelElement>) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <label
      className={`${styles.zone} ${dragOver ? styles.over : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <input
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFile(file);
        }}
      />
      <span className="label-caps">drop an image, or click to browse</span>
    </label>
  );
}
