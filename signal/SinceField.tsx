"use client";
import { useRef, useEffect } from "react";

type Props = {
  name?: string;
  defaultValueISO: string;
  inputClassName?: string;   // ← 追加
  withLabel?: boolean;       // （前回の提案を使っていない場合は省略可）
};


/**
 * datetime-local を使いつつ、submit時に ISO(Z) へ正規化して送るフィールド。
 * - 表示: ユーザー端末のローカル時刻で編集しやすい
 * - 送信: サーバ/APIが扱いやすい ISO (UTC) に変換
 */
export default function SinceField({
  name = "since",
  defaultValueISO,
  inputClassName = "",
  withLabel = true,
}: Props) {

  const formRef = useRef<HTMLFormElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const hiddenRef = useRef<HTMLInputElement | null>(null);

  // defaultValueISO -> datetime-local へ初期セット
  const toLocalInputValue = (iso: string) => {
    try {
      const d = new Date(iso);
      // yyyy-MM-ddTHH:mm 形式（秒未満は切り捨て）
      const pad = (n: number) => String(n).padStart(2, "0");
      const yyyy = d.getFullYear();
      const mm = pad(d.getMonth() + 1);
      const dd = pad(d.getDate());
      const hh = pad(d.getHours());
      const mi = pad(d.getMinutes());
      return `${yyyy}-${mm}-${dd}T${hh}:${mi}`;
    } catch {
      return "";
    }
  };

  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.value = toLocalInputValue(defaultValueISO);
    }
  }, [defaultValueISO]);

  useEffect(() => {
    if (!formRef.current) {
      let el: HTMLElement | null = inputRef.current ?? null;
      while (el && el.tagName !== "FORM") el = el.parentElement;
      formRef.current = el as HTMLFormElement | null;
    }
    const form = formRef.current;
    if (!form) return;

    const handleSubmit = () => {
      if (!inputRef.current || !hiddenRef.current) return;
      const localValue = inputRef.current.value;
      if (!localValue) {
        hiddenRef.current.disabled = true;
        return;
      }
      const dt = new Date(localValue);
      hiddenRef.current.value = dt.toISOString();
      hiddenRef.current.disabled = false;
    };

    form.addEventListener("submit", handleSubmit);
    return () => form.removeEventListener("submit", handleSubmit);
  }, []);

  return (
    <div>
      {withLabel && (
        <label className="block text-sm text-gray-600 dark:text-gray-300">
          Since（ローカル時刻で入力 → 送信時にISOへ）
        </label>
      )}
      <input
        ref={inputRef}
        type="datetime-local"
        className={
          `w-full border rounded-xl px-3 py-2 bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-700 ${inputClassName}`
        }
      />
      <input ref={hiddenRef} type="hidden" name={name} defaultValue={defaultValueISO} />
    </div>
  );
}
