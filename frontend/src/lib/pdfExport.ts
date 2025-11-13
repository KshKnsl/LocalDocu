import jsPDF from 'jspdf';
import { ChatDocument } from './chatStorage';

const setupPDF = () => {
  const pdf = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
  const pageWidth = pdf.internal.pageSize.getWidth(), pageHeight = pdf.internal.pageSize.getHeight();
  const margin = 25, contentWidth = pageWidth - 2 * margin;
  return { pdf, pageWidth, pageHeight, margin, contentWidth };
};

const addHeader = (pdf: jsPDF, title: string) => {
  const pageWidth = pdf.internal.pageSize.getWidth();
  pdf.setFontSize(9);
  pdf.setFont('helvetica', 'normal');
  pdf.setTextColor(0, 0, 0);
  pdf.text(title, 25, 15);
  pdf.setFontSize(8);
  pdf.text(new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }), pageWidth - 25, 15, { align: 'right' });
  pdf.setDrawColor(0, 0, 0);
  pdf.setLineWidth(0.3);
  pdf.line(25, 18, pageWidth - 25, 18);
};

const addFooters = (pdf: jsPDF, docTitle: string) => {
  const totalPages = pdf.internal.pages.length - 1, pageHeight = pdf.internal.pageSize.getHeight(), pageWidth = pdf.internal.pageSize.getWidth();
  for (let i = 1; i <= totalPages; i++) {
    pdf.setPage(i);
    if (i > 1) addHeader(pdf, docTitle);
    pdf.setDrawColor(0, 0, 0);
    pdf.setLineWidth(0.3);
    pdf.line(25, pageHeight - 20, pageWidth - 25, pageHeight - 20);
    pdf.setFontSize(8);
    pdf.setTextColor(0, 0, 0);
    pdf.text(`Page ${i} of ${totalPages}`, pageWidth / 2, pageHeight - 12, { align: 'center' });
  }
};

export async function exportChatToPDF(chat: ChatDocument) {
  const { pdf, pageWidth, pageHeight, margin, contentWidth } = setupPDF();
  let yPosition = margin + 10;

  // Title page
  pdf.setFontSize(20);
  pdf.setFont('helvetica', 'bold');
  pdf.setTextColor(0, 0, 0);
  pdf.text(chat.title || 'Document Report', margin, yPosition);
  yPosition += 10;
  
  pdf.setDrawColor(0, 0, 0);
  pdf.setLineWidth(0.5);
  pdf.line(margin, yPosition, pageWidth - margin, yPosition);
  yPosition += 10;

  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'normal');
  pdf.text(`Date: ${new Date(chat.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`, margin, yPosition);
  yPosition += 6;
  pdf.text(`Total Messages: ${chat.message_objects.length}`, margin, yPosition);
  yPosition += 15;

  // Attached documents
  if (chat.fileWithUrl?.length) {
    pdf.setFontSize(12);
    pdf.setFont('helvetica', 'bold');
    pdf.text('Attached Documents', margin, yPosition);
    yPosition += 7;
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    chat.fileWithUrl.forEach(file => {
      if (yPosition > pageHeight - 50) { pdf.addPage(); yPosition = margin + 30; }
      pdf.text(`• ${file.name}`, margin + 5, yPosition);
      yPosition += 6;
    });
    yPosition += 10;
  }

  // Conversation section
  pdf.setFontSize(12);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Conversation Transcript', margin, yPosition);
  yPosition += 2;
  pdf.setLineWidth(0.5);
  pdf.line(margin, yPosition, pageWidth - margin, yPosition);
  yPosition += 10;

  chat.message_objects.forEach((msg, i) => {
    if (yPosition > pageHeight - 50) { pdf.addPage(); yPosition = margin + 30; }
    
    // Author label
    pdf.setFontSize(11);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(0, 0, 0);
    pdf.text(msg.author === 'user' ? 'User:' : 'Assistant:', margin, yPosition);
    yPosition += 7;
    
    // Message content
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    const contentLines = pdf.splitTextToSize(msg.content.replace(/\*\*/g, '').replace(/\*/g, ''), contentWidth - 8);
    contentLines.forEach((line: string) => {
      if (yPosition > pageHeight - 35) { pdf.addPage(); yPosition = margin + 30; }
      pdf.text(line, margin + 4, yPosition);
      yPosition += 5;
    });
    yPosition += 3;

    // File attachments
    if (msg.files?.length) {
      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'italic');
      pdf.text(`Attachments: ${msg.files.map(f => f.name).join(', ')}`, margin + 4, yPosition);
      yPosition += 6;
    }

    // Citations
    if (msg.citations?.length) {
      yPosition += 3;
      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'bold');
      pdf.text('References:', margin + 4, yPosition);
      yPosition += 5;
      
      pdf.setFont('helvetica', 'normal');
      msg.citations.forEach((c, idx) => {
        if (yPosition > pageHeight - 35) { pdf.addPage(); yPosition = margin + 30; }
        const citText = `[${idx + 1}] ${c.source || 'Document'} - ${c.snippet?.substring(0, 80) || 'Reference'}${c.snippet && c.snippet.length > 80 ? '...' : ''}`;
        pdf.splitTextToSize(citText, contentWidth - 12).forEach((line: string) => {
          pdf.text(line, margin + 8, yPosition);
          yPosition += 4.5;
        });
      });
      yPosition += 3;
    }

    yPosition += 6;
    
    // Separator between messages
    if (i < chat.message_objects.length - 1) {
      pdf.setDrawColor(0, 0, 0);
      pdf.setLineWidth(0.2);
      pdf.line(margin + 10, yPosition, pageWidth - margin - 10, yPosition);
      yPosition += 8;
    }
  });

  addFooters(pdf, chat.title || 'Document Report');
  pdf.save(`${chat.title.replace(/[^a-z0-9]/gi, '_')}_${Date.now()}.pdf`);
}

export async function exportAllChatsToPDF(chats: ChatDocument[]) {
  const { pdf, pageWidth, pageHeight, margin, contentWidth } = setupPDF();
  let yPosition = margin + 10;

  // Cover page
  pdf.setFontSize(22);
  pdf.setFont('helvetica', 'bold');
  pdf.setTextColor(0, 0, 0);
  pdf.text('Document Archive', margin, yPosition);
  yPosition += 10;
  
  pdf.setDrawColor(0, 0, 0);
  pdf.setLineWidth(0.5);
  pdf.line(margin, yPosition, pageWidth - margin, yPosition);
  yPosition += 15;

  pdf.setFontSize(10);
  pdf.setFont('helvetica', 'normal');
  pdf.text(`Generated: ${new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`, margin, yPosition);
  yPosition += 6;
  pdf.text(`Total Documents: ${chats.length}`, margin, yPosition);
  yPosition += 6;
  pdf.text(`Total Messages: ${chats.reduce((sum, c) => sum + c.message_objects.length, 0)}`, margin, yPosition);
  yPosition += 20;

  // Table of contents
  pdf.setFontSize(14);
  pdf.setFont('helvetica', 'bold');
  pdf.text('Table of Contents', margin, yPosition);
  yPosition += 2;
  pdf.setLineWidth(0.5);
  pdf.line(margin, yPosition, pageWidth - margin, yPosition);
  yPosition += 10;

  chats.forEach((chat, idx) => {
    if (yPosition > pageHeight - 50) { pdf.addPage(); yPosition = margin + 30; }
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.text(`${idx + 1}. ${chat.title}`, margin + 5, yPosition);
    pdf.setFontSize(9);
    pdf.text(`${chat.message_objects.length} messages - ${new Date(chat.created_at).toLocaleDateString()}`, margin + 10, yPosition + 5);
    yPosition += 11;
  });

  // Documents
  pdf.addPage();
  yPosition = margin + 30;
  
  chats.forEach((chat, idx) => {
    if (idx > 0) { pdf.addPage(); yPosition = margin + 30; }
    
    // Document header
    pdf.setFontSize(14);
    pdf.setFont('helvetica', 'bold');
    pdf.setTextColor(0, 0, 0);
    pdf.text(`${idx + 1}. ${chat.title}`, margin, yPosition);
    yPosition += 7;
    
    pdf.setFontSize(10);
    pdf.setFont('helvetica', 'normal');
    pdf.text(`Date: ${new Date(chat.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}`, margin, yPosition);
    yPosition += 5;
    pdf.text(`Messages: ${chat.message_objects.length}`, margin, yPosition);
    
    if (chat.fileWithUrl?.length) {
      yPosition += 5;
      pdf.text(`Attachments: ${chat.fileWithUrl.length} file(s)`, margin, yPosition);
    }
    
    yPosition += 10;
    pdf.setLineWidth(0.3);
    pdf.line(margin, yPosition, pageWidth - margin, yPosition);
    yPosition += 10;

    chat.message_objects.forEach(msg => {
      if (yPosition > pageHeight - 45) { pdf.addPage(); yPosition = margin + 30; }
      
      pdf.setFontSize(10);
      pdf.setFont('helvetica', 'bold');
      pdf.text(msg.author === 'user' ? 'User:' : 'Assistant:', margin, yPosition);
      yPosition += 6;
      
      pdf.setFontSize(9);
      pdf.setFont('helvetica', 'normal');
      const lines = pdf.splitTextToSize(msg.content.replace(/\*\*/g, '').replace(/\*/g, ''), contentWidth - 8);
      lines.forEach((line: string) => {
        if (yPosition > pageHeight - 35) { pdf.addPage(); yPosition = margin + 30; }
        pdf.text(line, margin + 4, yPosition);
        yPosition += 4.5;
      });
      yPosition += 8;
    });
    yPosition += 5;
  });

  addFooters(pdf, 'Document Archive');
  pdf.save(`document_archive_${Date.now()}.pdf`);
}
