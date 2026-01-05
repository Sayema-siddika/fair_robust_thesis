# Day 27 Summary: Defense Presentation Creation

**Date:** December 6, 2024  
**Focus:** Create professional PowerPoint presentation for thesis defense

---

## Accomplishments

### 1. **Presentation Generation Script**
Created `experiments/27_create_presentation.py`:
- Automated PowerPoint generation using python-pptx library
- 20 slides: Title + 18 content + Q&A
- Three slide types: title, content, and image slides
- Professional design with custom styling

### 2. **Design Evolution**
Iterated through multiple design versions:

**Initial Version:**
- Plain white background
- Basic text layout
- No visual hierarchy

**Version 2 - Blue/Purple:**
- Blue-to-purple gradient title slide
- Light gray backgrounds
- Blue/green header bars
- Basic bullet points

**Final Version - Teal/Coral:**
- **Title slide:** Teal-to-cyan diagonal gradient (RGB: 13,148,136 → 6,182,212)
- **Background:** Warm amber/cream (RGB: 255,251,235)
- **Content slides:** Coral/orange header bars (RGB: 251,146,60)
- **Image slides:** Teal header bars (RGB: 20,184,166)
- **Cards:** White content cards with colored borders (2pt amber/teal)
- **Typography:**
  - Titles: Segoe UI Semibold (30pt)
  - Title slide: Segoe UI (48pt) / Segoe UI Light (26pt)
  - Body text: Calibri (19pt)
  - Bullet points: Small black squares (▪)
  - Line spacing: 1.3x for readability

### 3. **Presentation Structure**
20 slides organized for ~20-minute talk:

1. **Title Slide**
2-3. **Motivation & Research Questions** (3 min)
   - Fairness in ML problem
   - Real-world bias examples
   - Four research questions

4-5. **Method & Algorithm** (3 min)
   - Adaptive weighting formula: w_i = (c_i × r_i + ε)^(1/T)
   - 5-step iterative training algorithm

6-9. **Results** (6 min)
   - Slide 6: Perfect fairness text (German EO=0.0)
   - Slide 7: Fairness comparison chart (IMAGE)
   - Slide 8: Calibration degradation text (+388-756% ECE)
   - Slide 9: Reliability diagrams (IMAGE)

10-12. **Mechanism & Efficiency** (4 min)
   - Weight distribution evolution
   - Histogram visualization (IMAGE)
   - Computational efficiency (<2s training)

13-15. **Limitations** (3 min)
   - Dataset dependency
   - Calibration trade-off
   - Mitigation strategies

16-19. **Contributions & Future Work** (3 min)
   - Key contributions
   - Practical implications
   - Future research directions

20. **Q&A Slide**

### 4. **Supporting Materials**
Created `thesis/PRESENTATION_NOTES.md`:
- Complete speaking notes for all 20 slides
- Detailed timing breakdown (22 minutes total)
- 8 anticipated Q&A questions with comprehensive answers:
  - Why does calibration degrade?
  - Why does COMPAS fail?
  - Can we restore calibration?
  - How to choose temperature T?
  - Why not use existing fairness methods?
  - Real-world deployment concerns?
  - Ethical implications?
  - Future research priorities?
- Delivery tips and emphasis points
- Key phrases for smooth transitions

### 5. **Technical Solutions**
Fixed multiple technical issues:

**Image Path Issue:**
- Problem: Images not displaying in PowerPoint
- Root cause: Relative paths don't resolve in embedded PowerPoint files
- Solution: Convert to absolute paths using `Path("thesis") / img_path` then `str(img_path.absolute())`

**File Lock Issue:**
- Problem: PermissionError when regenerating while PowerPoint open
- Solution: User must close file before regeneration

**Library Installation:**
- Installed python-pptx 1.0.2 with dependencies:
  - lxml 6.0.2
  - XlsxWriter 3.2.9
  - Pillow 10.2.0

### 6. **Image Integration**
Successfully embedded 3 figures in presentation:
- `figures/fairness_comparison.png` (Slide 7)
- `figures/reliability_diagrams.png` (Slide 9)
- `figures/weight_distribution.png` (Slide 12)

All images display correctly with absolute paths.

---

## Design Specifications

### Color Palette
- **Primary gradient:** Teal (13,148,136) → Cyan (6,182,212)
- **Warm background:** Amber-50 (255,251,235)
- **Accent 1 (content):** Orange-400 (251,146,60)
- **Accent 2 (images):** Teal-500 (20,184,166)
- **Border (content):** Amber-400 (251,191,36)
- **Border (images):** Teal-400 (45,212,191)
- **Text primary:** Slate-800 (30,41,59)
- **Text secondary:** Slate-600 (71,85,105)

### Layout Dimensions
- **Slide size:** 10" × 7.5"
- **Header bars:** 9" wide × 0.75-0.9" tall
- **Content cards:** 8.6" × 5.3" (inset 0.7" from edges)
- **Margins:** 0.5-0.7" consistent spacing
- **Card borders:** 2pt width

### Typography Scale
- **Title slide main:** 48pt Segoe UI Bold
- **Title slide subtitle:** 26pt Segoe UI Light
- **Slide headers:** 26-30pt Segoe UI Semibold
- **Body text:** 19pt Calibri
- **Info text:** 15-18pt Segoe UI/Calibri

---

## Files Created/Modified

### New Files
1. `experiments/27_create_presentation.py` (457 lines)
2. `thesis/PRESENTATION_NOTES.md` (~6000 words)
3. `thesis/defense_presentation.pptx` (20 slides)

### Script Functions
- `add_title_slide()`: Creates gradient title slide
- `add_content_slide()`: Creates card-based content slides
- `add_image_slide()`: Creates image slides with captions

---

## Quality Metrics

### Presentation Stats
- **Total slides:** 20
- **Estimated duration:** 20 minutes (1 min/slide)
- **Image slides:** 3 (all working)
- **Content slides:** 16
- **File size:** ~2-3 MB (with embedded images)

### Design Principles Applied
✓ Consistent color scheme throughout  
✓ Professional typography hierarchy  
✓ Adequate white space and margins  
✓ High contrast for readability  
✓ Card-based modular design  
✓ Minimal but colorful aesthetic  
✓ Clean, modern corporate look  

---

## Next Steps (Day 28-30)

### Day 28: Thesis Review & Polish
- Read entire 71-page thesis PDF
- Check for typos and grammar errors
- Verify all citations and references
- Ensure figure references are correct
- Review mathematical notation consistency
- Final LaTeX compilation if changes made

### Day 29: Presentation Practice
- Practice 20-minute talk with timer
- Refine slide transitions and pacing
- Rehearse key talking points
- Practice Q&A responses
- Adjust timing if needed
- Record practice run (optional)

### Day 30: Final Defense Preparation
- Print thesis (if required)
- Print presentation slides (backup)
- Prepare backup slides for extra questions
- Review all materials one final time
- Mental preparation and confidence check
- Test presentation equipment

---

## Key Achievements

1. ✅ **Automated presentation generation** - Reproducible, customizable script
2. ✅ **Professional design** - Modern, colorful, minimalist aesthetic
3. ✅ **Working images** - All 3 figures display correctly
4. ✅ **Complete speaking notes** - 6000 words of guidance
5. ✅ **20-minute structure** - Proper pacing and flow
6. ✅ **Q&A preparation** - 8 anticipated questions answered

---

## Technical Notes

### Python-pptx Capabilities Used
- Custom slide layouts (blank layout #6)
- Background gradients and solid fills
- Textbox positioning and styling
- Shape creation (rectangles for cards/headers)
- Image embedding with absolute paths
- Font customization (name, size, color, bold)
- Text alignment and spacing
- Line and paragraph formatting

### Design Challenges Solved
1. **Gradient backgrounds** - Used `fill.gradient()` with angle and stops
2. **Card design** - Layered shapes (background → card → header → text)
3. **Color consistency** - RGB values from Tailwind CSS palette
4. **Professional bullets** - Unicode characters (▪ small square)
5. **Line spacing** - Set to 1.3x for optimal readability

---

## Lessons Learned

1. **File locking** - Must close PowerPoint before regenerating
2. **Absolute paths** - Required for embedded images in .pptx files
3. **Font availability** - Segoe UI and Calibri are Windows standard fonts
4. **Color harmony** - Warm backgrounds (amber) + cool accents (teal/coral) = balanced
5. **Iteration value** - Multiple design versions led to superior final product

---

## Time Investment

- Script development: ~2 hours
- Design iteration (3 versions): ~1.5 hours
- Speaking notes creation: ~1 hour
- Troubleshooting (images, fonts): ~0.5 hours
- **Total:** ~5 hours

---

## Status: ✅ COMPLETE

**Deliverables Ready:**
- ✅ 71-page thesis PDF (main.pdf)
- ✅ 20-slide defense presentation (defense_presentation.pptx)
- ✅ Complete speaking notes (PRESENTATION_NOTES.md)
- ✅ All figures embedded and working

**Remaining: Days 28-30** - Review, practice, final preparation

---

**Progress: 27/30 days (90% complete)**
