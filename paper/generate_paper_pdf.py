"""
Generate publication-quality preprint PDF of 'Plasticity Is All You Need?'
Author: Thomas Nauheimer (2026)
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas
from pathlib import Path


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#666666"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Plasticity Is All You Need? — Fast-Weight Adaptation")
            self.drawRightString(558, 750, "Thomas Nauheimer (2026)")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, "Preprint — Thomas Nauheimer — September 2026")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)
        self.restoreState()


def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1a1a2e'),
        alignment=1,
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#4a4e69'),
        alignment=1,
        spaceAfter=15
    )
    
    author_style = ParagraphStyle(
        'AuthorStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#222222'),
        alignment=1,
        spaceAfter=4
    )
    
    meta_style = ParagraphStyle(
        'MetaStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#555555'),
        alignment=1,
        spaceAfter=20
    )
    
    abstract_heading = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#1a1a2e'),
        alignment=1,
        spaceAfter=6
    )
    
    abstract_text = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#222222'),
        alignment=4
    )
    
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#162447'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#222222'),
        alignment=4,
        spaceAfter=8
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#111111'),
        leftIndent=15,
        spaceBefore=4,
        spaceAfter=6
    )
    
    story = []
    
    # Title & Metadata
    story.append(Spacer(1, 10))
    story.append(Paragraph("Plasticity Is All You Need?", title_style))
    story.append(Paragraph("A Testable Proposal for Persistent Fast-Weight Adaptation in Neural Architectures", subtitle_style))
    story.append(Paragraph("Thomas Nauheimer", author_style))
    story.append(Paragraph("Cognitive Systems Research &bull; Nierstein, Germany &bull; nauheimer.t@gmail.com", meta_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=15))
    
    # Abstract Box
    abstract_content = [
        Paragraph("Abstract", abstract_heading),
        Paragraph(
            "We investigate a frozen linear transformation augmented with persistent low-rank fast weights. "
            "Unlike external sidecar or retrieval-augmented architectures, the supplied <i>PlasticLinearProjected</i> adapter participates directly in a PyTorch "
            "model's forward computation. It applies explicit target-driven delta-rule updates and protects a specified input subspace "
            "span(Q) via orthogonal projection. Thin QR factorizations and a small-core SVD compress updates without explicitly reconstructing the dense fast-weight matrix. "
            "Synthetic linear-regression experiments demonstrate online adaptation, preservation of protected inputs within floating-point precision (&le; 10<sup>-16</sup>), "
            "unchanged base weights, and deterministic state serialization. These experiments establish implementation correctness in a testable, "
            "reproducible framework for continual fast-weight adaptation.",
            abstract_text
        )
    ]
    abstract_table = Table(
        [[abstract_content]],
        colWidths=[504]
    )
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8f9fa')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#e9ecef')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('RIGHTPADDING', (0,0), (-1,-1), 14),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 12))
    
    # Section 1
    story.append(Paragraph("1. Relationship to Existing Work", h1_style))
    story.append(Paragraph(
        "Fast weights, test-time adaptation, and trainable plasticity have established strong theoretical precedents in modern machine learning: "
        "<br/>&bull; <b>Fast Weights &amp; Linear Attention:</b> Ba et al. (2016) explored fast-weight memory for recurrent networks; Schlag et al. (2021) demonstrated that linearized self-attention mechanisms operate as fast-weight programmers. "
        "<br/>&bull; <b>Subspace &amp; Gradient Projection:</b> Orthogonal Weights Modification (OWM; Zeng et al., 2019) and Gradient Projection Memory (GPM; Saha et al., 2021) project gradient updates onto the orthogonal complement of protected feature subspaces to mitigate catastrophic forgetting. "
        "<br/>&bull; <b>Parameter vs. Activation Orthogonality (O-LoRA):</b> Wang et al. (2023; O-LoRA) enforce parameter-space orthogonality across discrete sequential tasks. In contrast, our method projects directly in activation space (onto the nullspace of historical activations) and operates continuously without task boundaries. "
        "<br/>&bull; <b>Test-Time Training (TTT):</b> Recent architectures such as TTT-Linear (Sun et al., 2024) and Titans (Behrouz et al., 2024) optimize hidden states via inference-time inner-loop updates. "
        "<br/>&bull; <b>Differentiable Plasticity &amp; EWC:</b> Miconi et al. (2018) optimized plastic connections via outer-loop meta-learning, while Kirkpatrick et al. (2017) utilized Fisher information matrices to penalize changes to important parameters.",
        body_style
    ))
    
    # Section 2
    story.append(Paragraph("2. Implemented Model and Mathematical Formulation", h1_style))
    story.append(Paragraph(
        "For row-batched inputs <i>X</i> &isin; &real;<sup>N &times; d<sub>in</sub></sup>, the forward mapping is defined as:",
        body_style
    ))
    story.append(Paragraph("<b>A(t) = U(t) V(t)<sup>T</sup></b>", code_style))
    story.append(Paragraph("<b>Y = X W<sub>slow</sub><sup>T</sup> + b + &gamma; (X V) U<sup>T</sup></b>", code_style))
    story.append(Paragraph(
        "where <i>W<sub>slow</sub></i> is the frozen base parameter matrix, <i>U</i> &isin; &real;<sup>d<sub>out</sub> &times; r</sup> "
        "and <i>V</i> &isin; &real;<sup>d<sub>in</sub> &times; r</sup> are dynamic low-rank factor buffers initialized to zero, and "
        "&gamma; &isin; [0, 1] scales adaptation energy. Given target activations <i>T</i>, error residual <i>E = T - Y</i>, "
        "and orthogonal subspace projector <i>P = I - Q Q<sup>T</sup></i>, the candidate update satisfies:",
        body_style
    ))
    story.append(Paragraph("<b>A<sub>cand</sub> = (1 - &lambda;) A + (&eta; &gamma; / N) E<sup>T</sup> X P</b>", code_style))
    story.append(Paragraph(
        "In deep language models, deriving verified intermediate target activations <i>T</i> remains an unsolved integration requirement; "
        "unverified model generations must not be treated automatically as correct learning targets.",
        body_style
    ))
    
    # Section 3
    story.append(Paragraph("3. Subspace Protection &amp; Invariance Properties", h1_style))
    story.append(Paragraph(
        "Let <i>Q</i> &isin; &real;<sup>d<sub>in</sub> &times; k</sup> denote an orthonormal basis for protected input features. "
        "If <i>A<sub>0</sub> Q = 0</i> and every successive update is right-projected by <i>P = I - Q Q<sup>T</sup></i>, "
        "then <i>A<sub>t</sub> Q = 0</i> holds identically in exact arithmetic. "
        "Consequently, the adapter leaves the linear layer output strictly invariant for all inputs in span(<i>Q</i>). "
        "<br/><br/><i>Critical Qualification:</i> Subspace invariance is an algebraic property of the projector <i>P</i> by construction; "
        "it protects only inputs residing strictly in span(<i>Q</i>). In deep autoregressive models, natural token activations "
        "frequently exhibit orthogonal components which undergo adaptation. Furthermore, every dimension allocated to <i>Q</i> "
        "reduces available rank capacity for novel associations.",
        body_style
    ))
    
    # Section 4
    story.append(Paragraph("4. Low-Rank Compression via Thin QR &amp; Core SVD", h1_style))
    story.append(Paragraph(
        "Representing candidate rank updates as <i>L R<sup>T</sup></i> with <i>m = r + N</i> columns: "
        "<br/>&bull; <i>L = [ &radic;(1-&lambda;) U, &radic;(&eta; &gamma;/N) E<sup>T</sup> ]</i> &isin; &real;<sup>d<sub>out</sub> &times; m</sup>"
        "<br/>&bull; <i>R = [ &radic;(1-&lambda;) P V, &radic;(&eta; &gamma;/N) P X<sup>T</sup> ]</i> &isin; &real;<sup>d<sub>in</sub> &times; m</sup>"
        "<br/>Thin QR factorizations <i>L = Q<sub>L</sub> R<sub>L</sub></i> and <i>R = Q<sub>R</sub> R<sub>R</sub></i> "
        "reduce the SVD to the small core matrix <i>R<sub>L</sub> R<sub>R</sub><sup>T</sup> &isin; &real;<sup>m &times; m</sup></i>. "
        "Total adaptation step complexity is <b>O((d<sub>in</sub> + d<sub>out</sub>) m<sup>2</sup> + m<sup>3</sup>)</b>. "
        "Crucially, the dense dimension product <i>d<sub>in</sub> &times; d<sub>out</sub></i> never appears in the computational complexity, "
        "as full weight matrices are never materialized. Forward overhead remains O(<i>N r (d<sub>in</sub> + d<sub>out</sub>)</i>).",
        body_style
    ))
    
    # Section 5: Experimental Results Table
    story.append(Paragraph("5. Empirical Validation &amp; Reproducibility", h1_style))
    story.append(Paragraph(
        "We evaluated <i>PlasticLinearProjected</i> across five independent random seeds (CPU, float64, 32-to-16 projection, rank 8, "
        "4 protected directions, 8 novel adaptation directions, 160 updates). The exact results below are reproducible via "
        "<code>python examples/reproduce_benchmark.py</code> (verified with <code>tests/test_projected_delta.py</code>):",
        body_style
    ))
    
    table_data = [
        ["Seed", "Held-out MSE Before", "Held-out MSE After", "Max Protected Change", "Invariance Check"],
        ["0", "0.254999", "2.74e-10", "3.47e-17", "Exact (AQ=0)"],
        ["1", "0.291153", "3.13e-10", "2.78e-17", "Exact (AQ=0)"],
        ["2", "0.227038", "2.44e-10", "2.78e-17", "Exact (AQ=0)"],
        ["3", "0.413457", "4.44e-10", "9.02e-17", "Exact (AQ=0)"],
        ["4", "0.501408", "5.38e-10", "3.47e-17", "Exact (AQ=0)"],
    ]
    t = Table(table_data, colWidths=[45, 125, 120, 130, 84])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#162447')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))
    
    # Section 6
    story.append(Paragraph("6. Production Integration Boundaries &amp; Conclusion", h1_style))
    story.append(Paragraph(
        "Integrating fast weights into production language models requires: (1) direct tensor contraction access rather than external API wrappers, "
        "(2) KV-cache coherence management during autoregressive decoding, and (3) systematic derivation of intermediate layer target activations <i>T</i>. "
        "The complete PyTorch implementation and unit test suite are released under the MIT License at: "
        "<code>https://github.com/nauheimervater/plastic-transformer</code>.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Successfully recompiled paper PDF to {output_path}")


if __name__ == "__main__":
    pdf_out = str(Path(__file__).parent / "Plasticity_Is_All_You_Need.pdf")
    build_pdf(pdf_out)
