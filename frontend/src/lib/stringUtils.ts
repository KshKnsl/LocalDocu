export const truncate = (str: string, max: number, suffix = "...") => str.length <= max ? str : str.substring(0, max - suffix.length) + suffix;
export const capitalize = (str: string) => str ? str[0].toUpperCase() + str.slice(1) : str;
export const generateId = (len = 7) => Math.random().toString(36).substring(2, 2 + len);
export const pluralize = (word: string, count: number, plural?: string) => count === 1 ? word : (plural || `${word}s`);
export const formatCount = (count: number, word: string, plural?: string) => `${count} ${pluralize(word, count, plural)}`;
