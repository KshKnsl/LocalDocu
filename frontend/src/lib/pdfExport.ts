import jsPDF from 'jspdf';
import { ChatDocument } from './chatStorage';

const setupPDF = () => {
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageWidth = pdf.internal.pageSize.getWidth(), pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 15, contentWidth = pageWidth - 2 * margin;
  return { pdf, pageWidth, pageHeight, margin, contentWidth };
};

const addFooters = (pdf: jsPDF) => {
  const totalPages = pdf.internal.pages.length - 1, pageHeight = pdf.internal.pageSize.getHeight(), pageWidth = pdf.internal.pageSize.getWidth();
  for (let i = 1; i <= totalPages; i++) {
    pdf.setPage(i);
    pdf.setFontSize(8);
    pdf.setTextColor(150, 150, 150);
    pdf.text(`Page ${i} of ${totalPages}`, pageWidth / 2, pageHeight - 10, { align: 'center' });
  }
};

export async function exportChatToPDF(chat: ChatDocument) {
  const { pdf, pageWidth, pageHeight, margin, contentWidth } = setupPDF();
  let yPosition = margin;

  pdf.setFontSize(18); pdf.setFont('helvetica', 'bold'); pdf.text(chat.title || 'Chat Export', margin, yPosition); yPosition += 10;
  pdf.setFontSize(10); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(100, 100, 100);
  pdf.text(`Created: ${new Date(chat.created_at).toLocaleString()}`, margin, yPosition); yPosition += 10;
  pdf.setDrawColor(200, 200, 200); pdf.line(margin, yPosition, pageWidth - margin, yPosition); yPosition += 10;

  if (chat.fileWithUrl?.length) {
    pdf.setFontSize(12); pdf.setFont('helvetica', 'bold'); pdf.setTextColor(0, 0, 0);
    pdf.text('Attached Files:', margin, yPosition); yPosition += 7; pdf.setFontSize(10); pdf.setFont('helvetica', 'normal');
    chat.fileWithUrl.forEach(file => { if (yPosition > pageHeight - margin) { pdf.addPage(); yPosition = margin; } pdf.text(`  • ${file.name}`, margin + 5, yPosition); yPosition += 5; });
    yPosition += 5;
  }
  pdf.setFontSize(12); pdf.setFont('helvetica', 'bold'); pdf.setTextColor(0, 0, 0); pdf.text('Conversation:', margin, yPosition); yPosition += 10;

  chat.message_objects.forEach((msg, i) => {
    if (yPosition > pageHeight - 40) { pdf.addPage(); yPosition = margin; }
    pdf.setFontSize(11); pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(msg.author === 'user' ? 37 : 34, msg.author === 'user' ? 99 : 197, msg.author === 'user' ? 235 : 94);
    pdf.text(msg.author === 'user' ? 'You:' : 'AI:', margin, yPosition); yPosition += 7;
    pdf.setFontSize(10); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(0, 0, 0);
    pdf.splitTextToSize(msg.content.replace(/\*\*/g, '').replace(/\*/g, ''), contentWidth).forEach((line: string) => {
      if (yPosition > pageHeight - margin) { pdf.addPage(); yPosition = margin; }
      pdf.text(line, margin, yPosition); yPosition += 5;
    });
    if (msg.files?.length) { yPosition += 3; pdf.setFontSize(9); pdf.setTextColor(100, 100, 100); pdf.text(`Attached: ${msg.files.map(f => f.name).join(', ')}`, margin + 5, yPosition); yPosition += 5; }
    if (msg.citations?.length) {
      yPosition += 3; pdf.setFontSize(9); pdf.setFont('helvetica', 'italic'); pdf.setTextColor(100, 100, 100); pdf.text('Sources:', margin + 5, yPosition); yPosition += 5;
      msg.citations.forEach((c, idx) => {
        if (yPosition > pageHeight - margin) { pdf.addPage(); yPosition = margin; }
        pdf.splitTextToSize(`[${idx + 1}] ${c.source || 'Unknown'} - ${c.snippet?.substring(0, 60) || ''}...`, contentWidth - 10).forEach((line: string) => { pdf.text(line, margin + 10, yPosition); yPosition += 4; });
      });
      yPosition += 2;
    }
    yPosition += 8;
    if (i < chat.message_objects.length - 1) { pdf.setDrawColor(230, 230, 230); pdf.line(margin, yPosition - 3, pageWidth - margin, yPosition - 3); }
  });
  addFooters(pdf);
  pdf.save(`${chat.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.pdf`);
}

export async function exportAllChatsToPDF(chats: ChatDocument[]) {
  const { pdf, pageWidth, pageHeight, margin, contentWidth } = setupPDF();
  let yPosition = margin;
  pdf.setFontSize(20); pdf.setFont('helvetica', 'bold'); pdf.text('Chat History Export', margin, yPosition); yPosition += 10;
  pdf.setFontSize(10); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(100, 100, 100);
  pdf.text(`Exported: ${new Date().toLocaleString()}`, margin, yPosition); yPosition += 7;
  pdf.text(`Total Chats: ${chats.length}`, margin, yPosition); yPosition += 15;
  
  chats.forEach((chat, idx) => {
    if (yPosition > pageHeight - 50) { pdf.addPage(); yPosition = margin; }
    if (idx > 0) { pdf.setDrawColor(100, 100, 100); pdf.setLineWidth(0.5); pdf.line(margin, yPosition, pageWidth - margin, yPosition); yPosition += 10; }
    pdf.setFontSize(14); pdf.setFont('helvetica', 'bold'); pdf.setTextColor(0, 0, 0); pdf.text(`${idx + 1}. ${chat.title}`, margin, yPosition); yPosition += 7;
    pdf.setFontSize(9); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(100, 100, 100); pdf.text(new Date(chat.created_at).toLocaleString(), margin, yPosition); yPosition += 10;
    chat.message_objects.forEach(msg => {
      if (yPosition > pageHeight - 40) { pdf.addPage(); yPosition = margin; }
      pdf.setFontSize(10); pdf.setFont('helvetica', 'bold');
      pdf.setTextColor(msg.author === 'user' ? 37 : 34, msg.author === 'user' ? 99 : 197, msg.author === 'user' ? 235 : 94);
      pdf.text(msg.author === 'user' ? 'You:' : 'AI:', margin + 5, yPosition); yPosition += 6;
      pdf.setFontSize(9); pdf.setFont('helvetica', 'normal'); pdf.setTextColor(0, 0, 0);
      pdf.splitTextToSize(msg.content.replace(/\*\*/g, '').replace(/\*/g, ''), contentWidth - 10).forEach((line: string) => {
        if (yPosition > pageHeight - margin) { pdf.addPage(); yPosition = margin; }
        pdf.text(line, margin + 5, yPosition); yPosition += 4.5;
      });
      yPosition += 6;
    });
    yPosition += 5;
  });
  addFooters(pdf);
  pdf.save(`all_chats_${Date.now()}.pdf`);
}
