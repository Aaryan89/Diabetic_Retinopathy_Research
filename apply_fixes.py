import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE

prs = Presentation('Diabetic_Retinopathy_Presentation.pptx')

# Delete Viva Notes from all slides
for slide in prs.slides:
    if slide.has_notes_slide:
        notes_frame = slide.notes_slide.notes_text_frame
        if notes_frame:
            new_paragraphs = []
            for p in notes_frame.paragraphs:
                if 'Viva:' not in p.text and 'viva:' not in p.text.lower():
                    new_paragraphs.append(p.text)
            notes_frame.text = '\n'.join(new_paragraphs)

# Slide 0: Title/Overview
for shape in prs.slides[0].shapes:
    if shape.has_text_frame:
        if "Comparative Evaluation of Nominal Softmax" in shape.text or "Comparative Evaluation:" in shape.text:
            shape.text_frame.text = "Comparative Evaluation: (A) Softmax, (B) CORAL, (C) Regression, and (D) CORN\nAccuracy Optimization: 3-Seed Ensemble + Label Smoothing + Post-Hoc Logit Adjustment"
            for p in shape.text_frame.paragraphs:
                p.font.size = Pt(14)
                p.font.color.rgb = RGBColor(203, 213, 225) # slate-300

# Function to remove tables from a slide
def remove_tables(slide):
    for shape in list(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.TABLE:
            shape._element.getparent().remove(shape._element)

# Slide 7: Literature Review Table Update
slide_7 = prs.slides[7]
remove_tables(slide_7)
x, y, cx, cy = Inches(0.5), Inches(1.5), Inches(12.33), Inches(5.0)
table_shape = slide_7.shapes.add_table(8, 4, x, y, cx, cy)
table = table_shape.table
columns = ['Ref & Authors', 'Reported Metrics', 'Major Limitation / Vulnerability', 'Our Decisive Advantage']
for i, col_name in enumerate(columns):
    table.cell(0, i).text = col_name

data = [
    ("[1] Khalifa et al.", "Acc: 97.9%", "Ignores 5-stage grading; cannot inform clinical referral", "Solves full 5-stage grading with 83.82% accuracy"),
    ("[2] Shakibania et al.", "Acc: 56.4%\nSev Recall: 36.84%", "Severe NPDR sensitivity collapsed to 36.84%", "+28.86% higher Severe Recall (65.7%)"),
    ("[3] Tymchenko et al.", "QWK: 0.841", "Relies on image blending rather than rank consistency", "Enforces mathematical rank consistency directly via CORN"),
    ("[4] Mardianta et al.", "Acc: 93.3%", "SMOTE creates blurry, non-anatomical artifacts", "Natural effective-number weighting (beta=0.9999)"),
    ("[5] Karthik et al.", "Acc: 95.8%\nSev Recall: 58%", "Simplified 5 classes into 3", "Retains all 5 grades and reaches 65.7% sev recall"),
    ("[6] Khokhar et al.", "QWK: 0.912", "Prohibitive compute latency (>350ms)", "Lightweight B0 (<12ms); mobile GPU deployable"),
    ("[7] Manoj & Bhosale", "Acc: 92.1%", "Unverified LLM referrals risk patient safety", "Mathematically bounds catastrophic errors to 3.8%-5.1%")
]

for row_idx, row_data in enumerate(data):
    for col_idx, cell_data in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.text = cell_data

for row in table.rows:
    for cell in row.cells:
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.font.size = Pt(12)
            paragraph.alignment = PP_ALIGN.LEFT

# Slide 9: Loss Formulations (Add Variant D)
slide_9 = prs.slides[9]
x, y, cx, cy = Inches(0.5), Inches(5.0), Inches(12.33), Inches(1.5)
tx = slide_9.shapes.add_textbox(x, y, cx, cy)
tf = tx.text_frame
tf.word_wrap = True
p = tf.add_paragraph()
p.text = "Variant D (CORN) & Accuracy Optimization:"
p.font.bold = True
p.font.size = Pt(18)
p.font.color.rgb = RGBColor(125, 211, 252)
p2 = tf.add_paragraph()
p2.text = "• CORN: Conditional probabilities, effective-number weighting (beta=0.9999), Soft-QWK loss.\n• Optimization: 3-seed ensemble (TTA), label smoothing (alpha=0.08), logit adjustment (tau)."
p2.font.size = Pt(14)
p2.font.color.rgb = RGBColor(255, 255, 255)

# Slide 10: Results (Remove old tables, add new single table)
slide_10 = prs.slides[10]
remove_tables(slide_10)

x, y, cx, cy = Inches(0.5), Inches(1.5), Inches(12.33), Inches(4.5)
table_shape = slide_10.shapes.add_table(8, 6, x, y, cx, cy)
table = table_shape.table
columns = ['Model Configuration', 'Accuracy', 'QWK', 'Severe NPDR Recall', 'Catastrophic Err (d>=2)', 'MAE']
for i, col_name in enumerate(columns):
    table.cell(0, i).text = col_name

data_res = [
    ("Variant A (Softmax)", "80.18%", "0.8724", "41.7%", "5.6%", "0.2673"),
    ("Variant B (CORAL)", "59.64%", "0.7273", "0.0%", "26.4%", "0.6964"),
    ("Variant C (Regression)", "75.64%", "0.8788", "38.4%", "3.8%", "0.2873"),
    ("Variant D (CORN)", "69.09%", "0.8482", "65.7%", "5.1%", "0.3764"),
    ("3-Seed Ensemble + TTA", "83.82%", "0.8859", "44.8%", "4.36%", "0.2200"),
    ("Ensemble + Logit (tau=0.5)", "82.18%", "0.8929", "51.7%", "-", "0.2291"),
    ("Ensemble + Logit (tau=1.5)", "74.91%", "0.8438", "69.0%", "-", "0.3364")
]

for row_idx, row_data in enumerate(data_res):
    for col_idx, cell_data in enumerate(row_data):
        cell = table.cell(row_idx + 1, col_idx)
        cell.text = cell_data

for row in table.rows:
    for cell in row.cells:
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(15, 23, 42)
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(255, 255, 255)
            paragraph.alignment = PP_ALIGN.LEFT

# Slide 13: Statistical Significance
slide_13 = prs.slides[13]
x, y, cx, cy = Inches(0.5), Inches(5.5), Inches(12.33), Inches(1.5)
tx = slide_13.shapes.add_textbox(x, y, cx, cy)
tf = tx.text_frame
tf.word_wrap = True
p = tf.add_paragraph()
p.text = "Variant D (CORN) vs. Variant B (CORAL):"
p.font.bold = True
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(125, 211, 252)
p2 = tf.add_paragraph()
p2.text = "ΔQWK = +0.1212 (95% CI: [+0.0924, +0.1504], p < 0.001). Wilcoxon signed-rank p = 9.35e-21. Statistically decisive."
p2.font.size = Pt(14)
p2.font.color.rgb = RGBColor(255, 255, 255)

# Slide 15: Limitations and Future Work
slide_15 = prs.slides[15]
for shape in slide_15.shapes:
    if shape.has_text_frame:
        if "Future work" in shape.text or "Inverse-frequency" in shape.text:
            shape.text_frame.clear()
            p = shape.text_frame.add_paragraph()
            p.text = "Future Work (Section 9.2)"
            p.font.bold = True
            p.font.size = Pt(18)
            p.font.color.rgb = RGBColor(125, 211, 252)
            
            f1 = shape.text_frame.add_paragraph()
            f1.text = "1. Multimodal Vision-Language Explanations: Integrate CORN with models like BiomedCLIP."
            f1.font.size = Pt(14)
            f1.font.color.rgb = RGBColor(255, 255, 255)
            
            f2 = shape.text_frame.add_paragraph()
            f2.text = "2. Multi-Center Clinical Validation: Generalization across Messidor-2, DDR, EyePACS."
            f2.font.size = Pt(14)
            f2.font.color.rgb = RGBColor(255, 255, 255)
            
            f3 = shape.text_frame.add_paragraph()
            f3.text = "3. Edge Deployment on Handheld Cameras: Quantize to INT8 using TensorRT."
            f3.font.size = Pt(14)
            f3.font.color.rgb = RGBColor(255, 255, 255)

# Delete Last Slide (Viva Cheatsheet)
# If the last slide has Viva in it
if len(prs.slides) > 0:
    last_slide = prs.slides[-1]
    has_viva = False
    for shape in last_slide.shapes:
        if shape.has_text_frame and "Viva" in shape.text:
            has_viva = True
            break
            
    if has_viva:
        rId = prs.slides._sldIdLst[-1].rId
        prs.part.drop_rel(rId)
        del prs.slides._sldIdLst[-1]

prs.save('Diabetic_Retinopathy_Presentation.pptx')
