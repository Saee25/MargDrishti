export function formatNumber(num) {
  if (num === null || num === undefined) return '-';
  return num.toLocaleString('en-US');
}

export function formatPercent(num) {
  if (num === null || num === undefined) return '-';
  return (num * 100).toFixed(2) + '%';
}

export function formatParams(num) {
  if (num === null || num === undefined) return '-';
  return (num / 1000000).toFixed(2) + ' M';
}

export function formatSize(num) {
  if (num === null || num === undefined) return '-';
  return num.toFixed(1) + ' MB';
}

export function formatTime(num) {
  if (num === null || num === undefined) return '-';
  return num.toFixed(1) + ' ms';
}
