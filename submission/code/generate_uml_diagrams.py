"""
Generate Publication-Quality System Architecture and UML Diagrams (StarUML Style)
for the Diabetic Retinopathy Deep Learning Decision Support Paper.
Diagrams:
  1. system_architecture.png
  2. uml_use_case_diagram.png
  3. uml_activity_diagram.png
  4. uml_class_diagram.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

RESULTS_DIR = os.path.join(".", "submission", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Color Palette
    bg_box = '#F8FAFC'
    border_col = '#334155'
    blue_box = '#2B5C8F'
    teal_box = '#0D9488'
    purple_box = '#7C3AED'
    orange_box = '#EA580C'

    # Title
    ax.text(7.0, 6.6, "System Architecture: Ordinal-Aware Diabetic Retinopathy Triage Framework",
            ha='center', va='center', fontsize=14, fontweight='bold', color='#0F172A')

    # Stage 1: Input & Preprocessing
    rect1 = patches.FancyBboxPatch((0.4, 1.2), 3.2, 4.8, boxstyle="round,pad=0.2",
                                   ec=border_col, fc='#EFF6FF', lw=1.5)
    ax.add_patch(rect1)
    ax.text(2.0, 5.7, "Stage 1: Preprocessing", ha='center', va='center', fontsize=11, fontweight='bold', color=blue_box)
    
    steps_s1 = [
        ("Raw Fundus Photograph\n(APTOS 2019 / 3,662 Images)", 4.8),
        ("Circular Bounding-Box Crop\n(Eliminates Black Border)", 3.7),
        ("CIE LAB Color Conversion\n& CLAHE on L-Channel", 2.6),
        ("Resolution Resizing\n(224 x 224 Normalized)", 1.5)
    ]
    for text, y in steps_s1:
        box = patches.FancyBboxPatch((0.6, y - 0.4), 2.8, 0.7, boxstyle="round,pad=0.1",
                                     ec='#93C5FD', fc='#FFFFFF', lw=1)
        ax.add_patch(box)
        ax.text(2.0, y - 0.05, text, ha='center', va='center', fontsize=8.5, color='#1E293B')

    # Arrow 1 -> 2
    ax.annotate('', xy=(4.1, 3.6), xytext=(3.6, 3.6),
                arrowprops=dict(facecolor=border_col, edgecolor=border_col, width=2, headwidth=8))

    # Stage 2: Feature Representation Backbone
    rect2 = patches.FancyBboxPatch((4.1, 1.2), 2.8, 4.8, boxstyle="round,pad=0.2",
                                   ec=border_col, fc='#F0FDFA', lw=1.5)
    ax.add_patch(rect2)
    ax.text(5.5, 5.7, "Stage 2: Feature Backbone", ha='center', va='center', fontsize=11, fontweight='bold', color=teal_box)

    steps_s2 = [
        ("EfficientNet-B0 (5.3M Params)\n(ImageNet Pretrained)", 4.6),
        ("Frozen Early Feature Blocks\n(features[:4] Edge Retention)", 3.4),
        ("Adaptive Average Pooling &\nSpatial Dropout (p = 0.3)", 2.2),
        ("Dense Representation Vector\n(d = 1280 Features)", 1.4)
    ]
    for text, y in steps_s2:
        box = patches.FancyBboxPatch((4.3, y - 0.4), 2.4, 0.65, boxstyle="round,pad=0.1",
                                     ec='#99F6E4', fc='#FFFFFF', lw=1)
        ax.add_patch(box)
        ax.text(5.5, y - 0.08, text, ha='center', va='center', fontsize=8.2, color='#1E293B')

    # Arrow 2 -> 3
    ax.annotate('', xy=(7.4, 3.6), xytext=(6.9, 3.6),
                arrowprops=dict(facecolor=border_col, edgecolor=border_col, width=2, headwidth=8))

    # Stage 3: Multi-Objective Prediction Heads
    rect3 = patches.FancyBboxPatch((7.4, 1.2), 3.2, 4.8, boxstyle="round,pad=0.2",
                                   ec=border_col, fc='#FAF5FF', lw=1.5)
    ax.add_patch(rect3)
    ax.text(9.0, 5.7, "Stage 3: Decision Heads", ha='center', va='center', fontsize=11, fontweight='bold', color=purple_box)

    heads = [
        ("Branch A: CORN Ordinal Head\nq_k = P(y > k | y > k-1)\n+ Differentiable Soft-QWK Loss", 4.5, '#7C3AED'),
        ("Branch B: 3-Seed Ensemble\nLabel Smoothing (alpha = 0.08)\n+ 4-Way Flip TTA Averaging", 3.1, '#2563EB'),
        ("Branch C: Logit Adjustment\nAdjusted: z - tau * log(pi_c)\n(Tau in [0.0, 1.5] Control)", 1.7, '#059669')
    ]
    for text, y, col in heads:
        box = patches.FancyBboxPatch((7.6, y - 0.5), 2.8, 0.95, boxstyle="round,pad=0.1",
                                     ec=col, fc='#FFFFFF', lw=1.2)
        ax.add_patch(box)
        ax.text(9.0, y - 0.02, text, ha='center', va='center', fontsize=8, color='#0F172A')

    # Arrow 3 -> 4
    ax.annotate('', xy=(11.1, 3.6), xytext=(10.6, 3.6),
                arrowprops=dict(facecolor=border_col, edgecolor=border_col, width=2, headwidth=8))

    # Stage 4: Clinical Expert Decision Support
    rect4 = patches.FancyBboxPatch((11.1, 1.2), 2.6, 4.8, boxstyle="round,pad=0.2",
                                   ec=border_col, fc='#FFF7ED', lw=1.5)
    ax.add_patch(rect4)
    ax.text(12.4, 5.7, "Stage 4: Clinical Triage", ha='center', va='center', fontsize=11, fontweight='bold', color=orange_box)

    triages = [
        ("Grade 0: No Retinopathy\n-> Routine 12-Mo Recall", 4.7, '#16A34A'),
        ("Grade 1: Mild NPDR\n-> 9-Mo Primary Follow-up", 3.8, '#0284C7'),
        ("Grade 2: Moderate NPDR\n-> 6-Mo Ophthalmology Review", 2.9, '#D97706'),
        ("Grade 3: Severe NPDR\n-> Urgent 48-Hr Referral", 2.0, '#DC2626'),
        ("Grade 4: Proliferative DR\n-> Emergency Vitreoretinal", 1.1, '#7F1D1D')
    ]
    for text, y, col in triages:
        box = patches.FancyBboxPatch((11.3, y - 0.35), 2.2, 0.65, boxstyle="round,pad=0.08",
                                     ec=col, fc='#FFFFFF', lw=1)
        ax.add_patch(box)
        ax.text(12.4, y - 0.02, text, ha='center', va='center', fontsize=7.6, fontweight='bold', color=col)

    out_path = os.path.join(RESULTS_DIR, "system_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

def generate_use_case_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    # Title
    ax.text(6.0, 7.1, "UML Use Case Diagram: Diabetic Retinopathy Clinical Triage System (StarUML Specification)",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')

    # System Boundary Box
    sys_box = patches.FancyBboxPatch((3.0, 0.6), 6.0, 6.0, boxstyle="square,pad=0",
                                     ec='#0284C7', fc='#F8FAFC', lw=2, ls='--')
    ax.add_patch(sys_box)
    ax.text(6.0, 6.3, "<<System>> DR Decision Support Platform",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#0369A1')

    # Actors Left: Clinic Technician, Patient
    def draw_actor(x, y, label):
        circle = patches.Circle((x, y + 0.4), 0.22, ec='#1E293B', fc='#E2E8F0', lw=1.5)
        ax.add_patch(circle)
        ax.plot([x, x], [y + 0.18, y - 0.25], color='#1E293B', lw=1.8) # spine
        ax.plot([x - 0.28, x + 0.28], [y + 0.05, y + 0.05], color='#1E293B', lw=1.8) # arms
        ax.plot([x, x - 0.25], [y - 0.25, y - 0.55], color='#1E293B', lw=1.8) # left leg
        ax.plot([x, x + 0.25], [y - 0.25, y - 0.55], color='#1E293B', lw=1.8) # right leg
        ax.text(x, y - 0.8, label, ha='center', va='center', fontsize=9, fontweight='bold', color='#0F172A')

    draw_actor(1.2, 5.0, "Screening\nTechnician")
    draw_actor(1.2, 2.0, "Screening\nPatient")

    # Actors Right: Ophthalmologist, Administrator
    draw_actor(10.8, 5.0, "Consulting\nOphthalmologist")
    draw_actor(10.8, 2.0, "System\nAdministrator")

    # Use Cases (Ellipses)
    use_cases = [
        ("UC-1: Upload Retinal Fundus Image", 5.6),
        ("UC-2: Run Circular Crop & CLAHE Preprocessing", 4.8),
        ("UC-3: Execute Ordinal Severity Grading (CORN)", 4.0),
        ("UC-4: Evaluate Distance-Bounded Confidence CIs", 3.2),
        ("UC-5: Review Sight-Threatening Triage Alert", 2.4),
        ("UC-6: Override / Confirm Clinical Stage", 1.6),
        ("UC-7: Generate ICO Referral Report", 0.9)
    ]

    uc_patches = {}
    for uc_name, y in use_cases:
        ellipse = patches.Ellipse((6.0, y), 4.6, 0.55, ec='#2563EB', fc='#FFFFFF', lw=1.2)
        ax.add_patch(ellipse)
        ax.text(6.0, y, uc_name, ha='center', va='center', fontsize=8.2, color='#1E293B')
        uc_patches[uc_name] = (6.0, y)

    # Actor-to-Use-Case Association Lines
    lines = [
        ((1.5, 4.8), (3.7, 5.6)), # Tech -> UC1
        ((1.5, 4.6), (3.7, 4.8)), # Tech -> UC2
        ((1.5, 4.4), (3.7, 4.0)), # Tech -> UC3
        ((1.5, 1.8), (3.7, 0.9)), # Patient -> UC7
        ((10.5, 4.8), (8.3, 4.0)), # Ophth -> UC3
        ((10.5, 4.6), (8.3, 3.2)), # Ophth -> UC4
        ((10.5, 4.4), (8.3, 2.4)), # Ophth -> UC5
        ((10.5, 4.2), (8.3, 1.6)), # Ophth -> UC6
        ((10.5, 4.0), (8.3, 0.9)), # Ophth -> UC7
        ((10.5, 2.0), (8.3, 3.2)), # Admin -> UC4
    ]
    for start, end in lines:
        ax.plot([start[0], end[0]], [start[1], end[1]], color='#64748B', lw=1.1, ls='-')

    out_path = os.path.join(RESULTS_DIR, "uml_use_case_diagram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

def generate_activity_diagram():
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    # Title
    ax.text(5.5, 5.58, "UML Activity Diagram: Clinical Screening & AI Triage Workflow (StarUML Specification)",
            ha='center', va='center', fontsize=11, fontweight='bold', color='#0F172A')

    # 3 Swimlanes
    lanes = [
        ('Primary Screening Clinic (Technician)', 0.2, 3.2, '#F8FAFC', '#38BDF8', '#0284C7'),
        ('Automated AI Diagnostic Engine (EfficientNet-B0)', 3.6, 4.2, '#F0FDF4', '#4ADE80', '#16A34A'),
        ('Clinical Triage & Specialist Referral', 8.0, 2.8, '#FFF7ED', '#FB923C', '#EA580C')
    ]

    for title, x, w, bg_col, border_col, hdr_col in lanes:
        lane_bg = patches.Rectangle((x, 0.2), w, 5.15, ec=border_col, fc=bg_col, lw=1.2, ls='--')
        ax.add_patch(lane_bg)
        hdr = patches.Rectangle((x, 5.05), w, 0.3, ec=border_col, fc=hdr_col, lw=1.2)
        ax.add_patch(hdr)
        ax.text(x + w/2, 5.2, title, ha='center', va='center', fontsize=8, fontweight='bold', color='#FFFFFF')

    def draw_action(x, y, w, h, text, ec='#2563EB', fc='#EFF6FF', fs=7.2, bold=False):
        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle='round,pad=0.08',
                                     ec=ec, fc=fc, lw=1.2)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, color='#0F172A',
                fontweight='bold' if bold else 'normal')

    def draw_arrow(x1, y1, x2, y2, label='', lpos=(0, 0)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#334155', edgecolor='#334155', width=1.2, headwidth=5, headlength=5))
        if label:
            lx = (x1 + x2)/2 + lpos[0]
            ly = (y1 + y2)/2 + lpos[1]
            ax.text(lx, ly, label, fontsize=6.8, color='#0369A1', fontweight='bold', ha='center', va='center')

    # --- LANE 1: Primary Screening Clinic ---
    start_c = patches.Circle((1.8, 4.75), 0.12, ec='#0F172A', fc='#0F172A')
    ax.add_patch(start_c)

    draw_arrow(1.8, 4.63, 1.8, 4.25)
    draw_action(1.8, 4.0, 2.6, 0.45, 'Acquire Retinal Fundus\nPhotograph (Handheld / Tabletop)')

    draw_arrow(1.8, 3.75, 1.8, 3.35)
    draw_action(1.8, 3.12, 2.6, 0.42, 'Automated Quality Check\n(Illumination & Focus Metric)')

    draw_arrow(1.8, 2.88, 1.8, 2.5)
    # Decision: Quality Pass?
    diamond_q = patches.Polygon([[1.8, 2.5], [2.2, 2.28], [1.8, 2.06], [1.4, 2.28]],
                                closed=True, ec='#0284C7', fc='#E0F2FE', lw=1.2)
    ax.add_patch(diamond_q)
    ax.text(1.8, 2.28, 'Quality\nPass?', ha='center', va='center', fontsize=6.6, fontweight='bold', color='#0369A1')

    # Loop back if Fail
    ax.plot([1.4, 0.45, 0.45, 0.65], [2.28, 2.28, 4.0, 4.0], color='#DC2626', lw=1.1, ls=':')
    draw_arrow(0.65, 4.0, 0.72, 4.0)
    ax.text(0.38, 3.15, '[No: Blurry / Glare]\nPrompt Recapture', fontsize=6.2, color='#DC2626', fontweight='bold', rotation=90, ha='center', va='center')

    # Forward to Lane 2 if Pass
    ax.plot([2.2, 4.0, 4.0], [2.28, 2.28, 4.5], color='#334155', lw=1.2)
    draw_arrow(4.0, 4.5, 4.25, 4.5)
    ax.text(2.85, 2.4, '[Yes: Valid]', fontsize=6.8, color='#16A34A', fontweight='bold')

    # --- LANE 2: AI Diagnostic Engine ---
    draw_action(5.7, 4.5, 2.8, 0.42, 'Preprocessing & CLAHE\n(Circular Crop, 224x224, CIE LAB)')

    draw_arrow(5.7, 4.28, 5.7, 3.88)
    draw_action(5.7, 3.65, 2.8, 0.42, 'Feature Extraction Backbone\n(EfficientNet-B0 -> Embedding phi(x))')

    draw_arrow(5.7, 3.42, 5.7, 3.05)
    # Decision / Branch: Screening Mode?
    diamond_m = patches.Polygon([[5.7, 3.05], [6.2, 2.82], [5.7, 2.59], [5.2, 2.82]],
                                closed=True, ec='#16A34A', fc='#DCFCE7', lw=1.2)
    ax.add_patch(diamond_m)
    ax.text(5.7, 2.82, 'Inference\nMode?', ha='center', va='center', fontsize=6.6, fontweight='bold', color='#15803D')

    # Mode A Left: CORN Ordinal
    ax.plot([5.2, 4.4, 4.4], [2.82, 2.82, 2.3], color='#334155', lw=1.1)
    draw_arrow(4.4, 2.3, 4.4, 2.05)
    ax.text(4.7, 2.92, '[CORN Mode]', fontsize=6.4, color='#15803D', fontweight='bold')
    draw_action(4.4, 1.82, 1.45, 0.42, 'CORN Ordinal\n(Soft-QWK Loss)', ec='#16A34A', fc='#FFFFFF', fs=6.8)

    # Mode B Right: 3-Seed Ensemble
    ax.plot([6.2, 7.0, 7.0], [2.82, 2.82, 2.3], color='#334155', lw=1.1)
    draw_arrow(7.0, 2.3, 7.0, 2.05)
    ax.text(6.7, 2.92, '[Ensemble]', fontsize=6.4, color='#15803D', fontweight='bold')
    draw_action(7.0, 1.82, 1.45, 0.42, '3-Seed + TTA\n(Logit Adj tau)', ec='#16A34A', fc='#FFFFFF', fs=6.8)

    # Join Bar (Synchronization bar)
    ax.plot([4.4, 4.4, 5.7], [1.6, 1.35, 1.35], color='#334155', lw=1.1)
    ax.plot([7.0, 7.0, 5.7], [1.6, 1.35, 1.35], color='#334155', lw=1.1)
    join_bar = patches.Rectangle((4.2, 1.3), 3.0, 0.08, ec='#0F172A', fc='#0F172A')
    ax.add_patch(join_bar)

    draw_arrow(5.7, 1.3, 5.7, 0.95)
    draw_action(5.7, 0.72, 3.2, 0.42, 'Enforce Distance Envelope & Triage Bounds\n(Suppresses |y - y_hat| >= 2 to <= 5.1%)', ec='#0284C7', fc='#E0F2FE', fs=6.8, bold=True)

    # Forward to Lane 3
    ax.plot([7.32, 8.4, 8.4], [0.72, 0.72, 4.5], color='#334155', lw=1.2)
    draw_arrow(8.4, 4.5, 8.65, 4.5)

    # --- LANE 3: Clinical Triage & Specialist Referral ---
    # Decision: Severity Grade?
    diamond_t = patches.Polygon([[9.4, 4.65], [9.85, 4.4], [9.4, 4.15], [8.95, 4.4]],
                                closed=True, ec='#EA580C', fc='#FFEDD5', lw=1.2)
    ax.add_patch(diamond_t)
    ax.text(9.4, 4.4, 'Predicted\nGrade?', ha='center', va='center', fontsize=6.6, fontweight='bold', color='#C2410C')

    # Branch 1: Severe/PDR (Grade 3, 4)
    ax.plot([9.85, 10.35, 10.35], [4.4, 4.4, 3.8], color='#334155', lw=1.1)
    draw_arrow(10.35, 3.8, 10.35, 3.55)
    ax.text(10.25, 4.52, '[Grade 3, 4]', fontsize=6.3, color='#DC2626', fontweight='bold')
    draw_action(10.35, 3.32, 1.15, 0.42, 'Urgent Referral\n(<48h to 2w)', ec='#DC2626', fc='#FEF2F2', fs=6.6, bold=True)

    # Branch 2: Moderate (Grade 2)
    draw_arrow(9.4, 4.15, 9.4, 3.55)
    ax.text(9.4, 3.85, '[Grade 2]', fontsize=6.3, color='#D97706', fontweight='bold', ha='center')
    draw_action(9.4, 3.32, 1.15, 0.42, '6-Mo Review\n+ OCT Scan', ec='#D97706', fc='#FFFBEB', fs=6.6)

    # Branch 3: No DR / Mild (Grade 0, 1)
    ax.plot([8.95, 8.45, 8.45], [4.4, 4.4, 3.8], color='#334155', lw=1.1)
    draw_arrow(8.45, 3.8, 8.45, 3.55)
    ax.text(8.55, 4.52, '[Grade 0, 1]', fontsize=6.3, color='#16A34A', fontweight='bold')
    draw_action(8.45, 3.32, 1.15, 0.42, '12-Mo Recall\nSurveillance', ec='#16A34A', fc='#F0FDF4', fs=6.6)

    # Merge Branches
    ax.plot([8.45, 8.45, 9.4], [3.1, 2.75, 2.75], color='#334155', lw=1.1)
    ax.plot([10.35, 10.35, 9.4], [3.1, 2.75, 2.75], color='#334155', lw=1.1)
    draw_arrow(9.4, 3.1, 9.4, 2.75)

    merge_t = patches.Rectangle((8.35, 2.7), 2.1, 0.07, ec='#0F172A', fc='#0F172A')
    ax.add_patch(merge_t)

    draw_arrow(9.4, 2.7, 9.4, 2.3)
    draw_action(9.4, 2.05, 2.4, 0.45, 'Compile Structured Report\n& Transmit to PACS Database', ec='#475569', fc='#FFFFFF', fs=6.8)

    draw_arrow(9.4, 1.8, 9.4, 1.35)
    draw_action(9.4, 1.1, 2.4, 0.42, 'Notify Care Team & Patient\n(SMS / Telemedicine Portal)', ec='#475569', fc='#F8FAFC', fs=6.8)

    # Final Bullseye State Node
    draw_arrow(9.4, 0.88, 9.4, 0.58)
    final_out = patches.Circle((9.4, 0.45), 0.13, ec='#0F172A', fc='#FFFFFF', lw=1.4)
    final_in  = patches.Circle((9.4, 0.45), 0.08, ec='#0F172A', fc='#0F172A')
    ax.add_patch(final_out)
    ax.add_patch(final_in)

    out_path = os.path.join(RESULTS_DIR, "uml_activity_diagram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

def generate_class_diagram():
    fig, ax = plt.subplots(figsize=(13, 8), dpi=300)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis('off')

    ax.text(6.5, 7.7, "UML Class Diagram: Object-Oriented Framework Architecture (StarUML Specification)",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#0F172A')

    def draw_class(x, y, w, h, name, attrs, methods, header_color='#1E40AF'):
        # Class container
        box = patches.Rectangle((x, y), w, h, ec='#334155', fc='#FFFFFF', lw=1.4)
        ax.add_patch(box)
        # Class Name header
        hdr = patches.Rectangle((x, y + h - 0.45), w, 0.45, ec='#334155', fc=header_color, lw=1.4)
        ax.add_patch(hdr)
        ax.text(x + w/2, y + h - 0.22, name, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF')

        # Attributes
        sep_y = y + h - 0.45 - (len(attrs)*0.24 + 0.15)
        ax.plot([x, x + w], [sep_y, sep_y], color='#94A3B8', lw=1)

        curr_y = y + h - 0.65
        for attr in attrs:
            ax.text(x + 0.1, curr_y, attr, fontsize=7.2, color='#1E293B', fontfamily='monospace')
            curr_y -= 0.24

        # Methods
        curr_y = sep_y - 0.25
        for m in methods:
            ax.text(x + 0.1, curr_y, m, fontsize=7.2, color='#047857', fontfamily='monospace')
            curr_y -= 0.24

    # Class 1: FundusImage
    draw_class(0.5, 4.6, 3.4, 2.6, "FundusImage",
               ["- id_code: String", "- raw_rgb: np.ndarray", "- clinical_grade: int", "- quality_score: float"],
               ["+ load_from_disk(): void", "+ get_aspect_ratio(): float", "+ validate_resolution(): bool"])

    # Class 2: CLAHEPreprocessor
    draw_class(4.6, 4.6, 3.8, 2.6, "RetinalPreprocessor",
               ["- clip_limit: float = 2.0", "- tile_grid: tuple = (8, 8)", "- target_dim: tuple = (224, 224)"],
               ["+ crop_black_borders(): np.ndarray", "+ apply_lab_clahe(): np.ndarray", "+ normalize_imagenet(): Tensor"])

    # Class 3: DRModel (EfficientNet-B0)
    draw_class(9.1, 4.6, 3.4, 2.6, "DRModel",
               ["- backbone_name: String", "- num_classes: int = 5", "- freeze_early: bool", "- dropout_rate: float = 0.3"],
               ["+ forward(x: Tensor): Tensor", "+ extract_features(): Tensor", "+ predict(x, tta=True): tuple"])

    # Class 4: CornOrdinalHead
    draw_class(0.5, 0.8, 3.6, 3.0, "CornOrdinalHead",
               ["- num_tasks: int = 4", "- weights: Tensor", "- conditional_logits: Tensor"],
               ["+ forward(features): Tensor", "+ compute_corn_loss(): Tensor", "+ predict_cum_probs(): Tensor", "+ predict_discrete_probs(): Tensor"])

    # Class 5: SoftQWKLoss
    draw_class(4.8, 0.8, 3.6, 3.0, "SoftQWKLoss",
               ["- num_classes: int = 5", "- weight_matrix: Tensor", "- eps: float = 1e-7"],
               ["+ compute_observed_error(): Tensor", "+ compute_expected_error(): Tensor", "+ forward(probs, targets): Tensor"])

    # Class 6: TriageDecisionSupport
    draw_class(9.1, 0.8, 3.4, 3.0, "TriageDecisionSupport",
               ["- ico_rules: dict", "- tau_threshold: float = 0.5", "- referral_status: String"],
               ["+ map_grade_to_referral(): String", "+ evaluate_safety_envelope(): bool", "+ generate_clinical_report(): String"])

    # Connecting arrows
    def draw_assoc(x1, y1, x2, y2, label=""):
        ax.plot([x1, x2], [y1, y2], color='#475569', lw=1.2)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.12, label, fontsize=7.5, color='#475569', ha='center')

    draw_assoc(3.9, 5.9, 4.6, 5.9, "preprocesses")
    draw_assoc(8.4, 5.9, 9.1, 5.9, "feeds into")
    draw_assoc(2.3, 4.6, 2.3, 3.8, "parameterizes")
    draw_assoc(6.6, 4.6, 6.6, 3.8, "regularizes")
    draw_assoc(10.8, 4.6, 10.8, 3.8, "routes to")

    out_path = os.path.join(RESULTS_DIR, "uml_class_diagram.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_system_architecture()
    generate_use_case_diagram()
    generate_activity_diagram()
    generate_class_diagram()
