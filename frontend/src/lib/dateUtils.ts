const toDate = (d: string | Date) => typeof d === "string" ? new Date(d) : d;

export const formatDate = (date: string | Date, opts?: Intl.DateTimeFormatOptions) => toDate(date).toLocaleDateString(undefined, opts);
export const formatDateTime = (date: string | Date, opts?: Intl.DateTimeFormatOptions) => toDate(date).toLocaleString(undefined, opts);
export const formatShortDate = (date: string | Date) => formatDate(date, { month: "short", day: "numeric", year: "numeric" });

export const getRelativeTime = (date: string | Date) => {
  const secs = Math.floor((Date.now() - toDate(date).getTime()) / 1000);
  const mins = Math.floor(secs / 60), hrs = Math.floor(mins / 60), days = Math.floor(hrs / 24);
  if (secs < 60) return "just now";
  if (mins < 60) return `${mins} min${mins > 1 ? "s" : ""} ago`;
  if (hrs < 24) return `${hrs} hr${hrs > 1 ? "s" : ""} ago`;
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  return formatShortDate(date);
};
