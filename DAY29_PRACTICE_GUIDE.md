# Day 29: Presentation Practice Guide

**Date:** December 6, 2024  
**Task:** Master 20-minute defense presentation  
**Goal:** Confident, clear, professional delivery

---

## Practice Schedule

### Session 1: Solo Run-Through (30 minutes)
1. **Full presentation** with timer (target: 20 minutes)
2. Record actual time per slide
3. Identify problem areas (rushed sections, unclear explanations)
4. Note: Don't stop for mistakes - complete full run

### Session 2: Slow Practice (45 minutes)
1. **Slide-by-slide** deep practice
2. Rehearse key phrases and transitions
3. Practice difficult pronunciations ("equalized odds", "calibration", "disparities")
4. Memorize opening and closing statements

### Session 3: Timed Run (25 minutes)
1. **Full presentation** with strict timing
2. Use countdown timer (20 minutes visible)
3. Hit timing targets: slides 1-5 (5 min), 6-15 (12 min), 16-20 (3 min)
4. Record and review if possible

### Session 4: Q&A Practice (30 minutes)
1. Practice answers to 8 anticipated questions (see below)
2. Time each answer (target: 1-2 minutes)
3. Practice saying "I don't know" gracefully for unexpected questions
4. Rehearse redirecting to strengths

---

## Slide-by-Slide Timing Plan

| Slide | Topic | Target Time | Key Points |
|-------|-------|-------------|------------|
| 1 | Title | 0:30 | Introduce yourself, thesis title, thank committee |
| 2 | Motivation | 1:00 | Real-world bias examples (COMPAS, Amazon) |
| 3 | Research Questions | 1:30 | Four RQs clearly stated |
| 4 | Method Overview | 1:00 | Weight formula, explain variables |
| 5 | Algorithm | 1:30 | 5-step process, emphasize simplicity |
| 6 | Perfect Fairness Result | 1:00 | German EO=0.0, Adult 68.7% improvement |
| 7 | Fairness Chart (IMAGE) | 0:45 | Point to bars, highlight German |
| 8 | Calibration Trade-off | 1:00 | +388-756% ECE degradation |
| 9 | Reliability Diagrams (IMAGE) | 0:45 | Show calibration collapse visually |
| 10 | Accuracy Trade-off | 0:45 | Minimal loss (-0.9% to -1.8%) |
| 11 | Mechanism Explanation | 1:15 | Why upweighting confident samples works |
| 12 | Weight Distribution (IMAGE) | 0:45 | Iteration 1 vs 5 comparison |
| 13 | Efficiency Results | 1:00 | <2s training, zero inference overhead |
| 14 | Dataset Dependency | 1:00 | German perfect, Adult good, COMPAS limited |
| 15 | Calibration Limitations | 1:00 | When NOT to use this method |
| 16 | Key Contributions | 1:00 | Novel formula, perfect fairness, trade-off quantification |
| 17 | Practical Implications | 1:00 | Loan approval (yes), medical diagnosis (no) |
| 18 | Future Work | 0:45 | Calibration restoration, theoretical analysis |
| 19 | Conclusion | 0:45 | Summarize achievements |
| 20 | Questions | 0:15 | "Thank you, happy to answer questions" |

**TOTAL: ~19 minutes** (1 minute buffer for transitions)

---

## Critical Phrases to Memorize

### Opening (Slide 1)
> "Good morning/afternoon. My name is [Your Name], and today I'll present my bachelor's thesis on fair and robust training with adaptive sample weighting. Thank you to the committee for your time."

### Thesis Statement (Slide 2)
> "The central problem I address is that machine learning systems deployed in high-stakes domains—such as criminal justice and hiring—often perpetuate societal biases, leading to unfair outcomes for protected demographic groups."

### Main Achievement (Slide 6)
> "Our key result is achieving **perfect equalized odds**—that is, **zero fairness violations**—on the German Credit dataset. To my knowledge, this is the **first demonstration** of an in-processing method achieving EO equals zero point zero on real-world data."

### Trade-off Discovery (Slide 8)
> "However, this fairness improvement comes at a cost: calibration degrades by **388 to 756 percent** as measured by expected calibration error. This is the **first empirical quantification** of how in-processing fairness interventions affect calibration quality."

### Mechanism Insight (Slide 11)
> "Counterintuitively, our method **upweights samples the model already handles well**—confident correct predictions—rather than focusing on errors like traditional boosting. This works because disadvantaged groups have lower confidence even for correct predictions, and amplifying their successes rebalances group-wise performance."

### Practical Guidance (Slide 17)
> "For applications that use **binary decisions**—like loan approval or hiring screening—our method is suitable because calibration doesn't matter, only the accept/reject outcome. However, for applications requiring **probability estimates**—like medical risk assessment—the calibration degradation is unacceptable."

### Closing (Slide 19)
> "In conclusion, this thesis demonstrates that perfect fairness is achievable in practice using a simple iterative weighting mechanism, though practitioners must carefully evaluate the fairness-calibration trade-off for their specific use case. Thank you."

---

## Anticipated Questions & Prepared Answers

### Q1: "Why does calibration degrade so severely?"
**Answer (90 seconds):**
> "Great question. The calibration degradation is a direct consequence of our weighting formula. By upweighting confident correct predictions with the formula w = (confidence × correctness)^(1/T), we're telling the optimizer to focus on samples where the model is already very sure and right. This creates a reinforcement loop: the model becomes even more confident in regions it already understands well, pushing probabilities toward extremes—zero or one—rather than moderate values like 0.6 or 0.7. Calibration requires probabilities to match true frequencies, so when we push everything to extremes, we break that alignment. The mechanism is: extreme confidence → overconfidence → poor calibration. This is why reliability diagrams in Figure 9 show the curve deviating dramatically from the diagonal after our intervention."

### Q2: "Why does your method fail on COMPAS but succeed on German Credit?"
**Answer (90 seconds):**
> "This is one of the limitations I acknowledge in the thesis. I hypothesize it's related to **group imbalance structure**. German Credit has highly imbalanced groups: only 15% are under age 25 versus 85% over 25. This creates large confidence differences between groups, which our method exploits by amplifying the confident correct predictions from the disadvantaged group. COMPAS, however, has nearly balanced groups: 51% African-American, 49% Caucasian. With similar group sizes, there's less confidence asymmetry to leverage. Additionally, German has a smaller sample size (n=1,000) which may allow perfect fairness to be achieved more easily than COMPAS (n=6,000). However, I didn't formally prove this relationship—that's valuable future work to predict which datasets will succeed based on imbalance ratio and base rate differences."

### Q3: "Can you restore calibration while maintaining fairness?"
**Answer (60 seconds):**
> "Yes, theoretically. A promising approach is **post-hoc recalibration** using temperature scaling. After achieving perfect fairness with our method, we could apply Platt scaling or isotonic regression to rescale the probabilities without changing the decision boundaries. This would restore calibration—bringing ECE back down—while keeping the fairness gains because the threshold-based predictions (y-hat > 0.5) remain unchanged. I didn't implement this in my thesis due to scope limitations, but it's the top priority for future work. The key insight is that calibration and decision accuracy are somewhat orthogonal: you can fix probabilities post-hoc without affecting the binary outcomes."

### Q4: "How did you choose temperature T=1.0 for German Credit?"
**Answer (75 seconds):**
> "I used **grid search** over T in {0.5, 1.0, 2.0, 5.0} during my Day 10 uncertainty weighting experiments. Lower temperature (T=0.5) creates sharper weight distributions—very high weights for confident correct samples, very low for everything else. Higher temperature (T=5.0) creates smoother distributions. For German Credit, T=1.0 provided the best balance: sharp enough to drive fairness to zero in just 4 iterations, but not so sharp that it caused numerical instability or overfitting. Interestingly, different datasets preferred different temperatures: Adult worked best at T=1.0, while COMPAS showed slightly better results at T=2.0. This suggests the optimal temperature is dataset-dependent, likely correlated with group imbalance ratio. A more sophisticated approach would be to tune T via cross-validation or adapt it per-iteration, which I leave for future work."

### Q5: "Why not use existing fairness methods like Zafar et al.'s constraints?"
**Answer (75 seconds):**
> "Constrained optimization methods like Zafar et al. 2017 have two main challenges: **convergence issues** and **complexity**. Fairness constraints create non-convex optimization landscapes, making it hard to guarantee the solver will find the global optimum. In my experiments, constrained methods achieved EO around 0.02 to 0.05—good, but not perfect. My method, by contrast, achieves EO=0.000 through simple iterative reweighting of standard logistic regression, which has well-understood convex properties. Additionally, implementation complexity differs: constrained optimization requires custom Lagrangian solvers (150+ lines of code), while my method is 10 lines—just weight computation plus sklearn's LogisticRegression with sample_weight parameter. This simplicity aids deployment, debugging, and stakeholder explanation. However, I acknowledge constrained methods offer theoretical guarantees I don't provide."

### Q6: "What are the ethical implications of perfect fairness?"
**Answer (90 seconds):**
> "This is a profound question. Achieving perfect fairness—EO equals zero—means true positive rates and false positive rates are exactly equal across demographic groups. This aligns with the **equality of opportunity** principle: people should have equal chances regardless of protected attributes. However, there are nuances. First, perfect **statistical fairness** doesn't guarantee **individual fairness**—two similar people might still receive different outcomes. Second, the **calibration trade-off** raises ethical concerns: if probabilities become meaningless, can we ethically deploy such systems in high-stakes settings? I argue **context matters**: for binary decisions where only the outcome matters (loan approval), perfect fairness is ethically desirable. But for probability-dependent settings (medical diagnosis), degraded calibration could harm individual decision-making. The ethical imperative is **transparency**: practitioners must inform stakeholders about this trade-off and choose methods matching their values—fairness-first or calibration-first."

### Q7: "How does this scale to larger datasets or deep learning?"
**Answer (75 seconds):**
> "For **larger datasets**, my scalability analysis in Section 4.4 shows **linear O(n) scaling**: training time increases proportionally with sample size, which is acceptable. I tested up to 45,000 samples (Adult dataset) with training times under 2 seconds. For **deep learning**, the weighting mechanism is architecture-agnostic—neural networks support sample weights just like logistic regression. However, two challenges arise: First, **computational cost** increases significantly with deep models (minutes instead of seconds per iteration). Second, **calibration degradation may worsen** because neural networks are already poorly calibrated (Guo et al. 2017), and our method could exacerbate this. I focused on logistic regression for my BSc thesis to isolate the weighting mechanism's effects, but extending to deep learning is valuable future work, potentially requiring architectural modifications like fairness-aware batch normalization."

### Q8: "What would you do differently if you could redo this research?"
**Answer (90 seconds):**
> "Three things. First, **theoretical analysis**: I would derive convergence guarantees—prove that Algorithm 1 converges to EO=0 under certain conditions, like convex loss and bounded gradients. This would strengthen the contribution from empirical-only to theory-backed. Second, **more datasets**: I used 3 benchmarks, but evaluating on 10+ datasets from UCI repository would better characterize when the method succeeds versus fails. I hypothesized group imbalance predicts effectiveness, but didn't formally test this—a regression analysis of (imbalance ratio, base rate difference) versus EO reduction would be valuable. Third, **calibration restoration**: I'd implement post-hoc temperature scaling to demonstrate that perfect fairness + good calibration is achievable simultaneously, removing the trade-off limitation. These extensions would elevate the work from a strong BSc thesis to a publishable conference paper. However, given the 30-day timeline, I'm satisfied with what I achieved."

---

## Delivery Tips

### Body Language
✅ **Stand confidently** - Don't lean on podium  
✅ **Make eye contact** - Rotate between committee members  
✅ **Use hand gestures** - Point to charts when explaining  
✅ **Smile occasionally** - Show enthusiasm for your work  
❌ Avoid: Fidgeting, reading slides verbatim, turning back to audience

### Voice Control
✅ **Project clearly** - Speak to back of room  
✅ **Vary pace** - Slow down for key results (EO=0.0)  
✅ **Pause strategically** - After major claims, let them sink in  
✅ **Emphasize numbers** - "Three hundred **eighty-eight** percent"  
❌ Avoid: Monotone, too fast, mumbling

### Handling Nerves
- **Deep breath** before starting  
- **Pause and sip water** if flustered  
- **It's okay to say "That's a great question, let me think..."**  
- **Redirect tough questions** to thesis strengths: "While I didn't explore that fully, what I *did* find is..."

### Time Management During Talk
- **Glance at clock** every 5 slides (don't stare)  
- **If running over:** Skip details on slides 14-15 (dataset dependency, limitations)  
- **If running under:** Add examples on slide 2 (bias stories), elaborate on slide 11 (mechanism)  
- **Buffer strategy:** Slides 17-19 can be compressed to 1 minute each if needed

---

## Equipment Checklist

### Physical Setup
- [ ] Laptop fully charged + power adapter  
- [ ] Presentation clicker (test batteries)  
- [ ] HDMI/VGA adapter (test connection)  
- [ ] Backup presentation on USB drive  
- [ ] Water bottle  
- [ ] Printed slide deck (optional, for notes)

### Digital Prep
- [ ] Close all other applications (email, Slack, notifications OFF)  
- [ ] Set PowerPoint to Presenter View (notes + timer visible to you)  
- [ ] Test animations/transitions work on presentation computer  
- [ ] Verify images display correctly  
- [ ] Set screen resolution to match projector

---

## Practice Session Template

### Before Each Practice Run:
1. Set timer for 20 minutes  
2. Clear your mind (30 seconds silence)  
3. Start with opening statement verbatim  
4. Pretend committee is in the room

### During Practice:
- **Don't restart** if you make mistakes - keep going  
- **Note timestamp** when you finish each slide  
- **Mark unclear sections** to revise  
- **Record yourself** (phone video) to review

### After Each Practice:
1. Check total time (goal: 18-22 minutes)  
2. Identify 3 things done well  
3. Identify 3 things to improve  
4. Rest 10 minutes, then practice those 3 improvements

---

## Self-Assessment Rubric

Rate yourself 1-5 after each practice (goal: all 4+ by final run):

| Criterion | Run 1 | Run 2 | Run 3 | Target |
|-----------|-------|-------|-------|--------|
| Timing (18-22 min) | ___ | ___ | ___ | 4+ |
| Clarity of speech | ___ | ___ | ___ | 4+ |
| Confidence level | ___ | ___ | ___ | 4+ |
| Eye contact | ___ | ___ | ___ | 4+ |
| Smooth transitions | ___ | ___ | ___ | 4+ |
| Explaining figures | ___ | ___ | ___ | 4+ |
| Handling Q&A | ___ | ___ | ___ | 4+ |
| Energy/enthusiasm | ___ | ___ | ___ | 4+ |

**Scoring:**
- 1 = Needs major improvement
- 2 = Needs some work
- 3 = Acceptable
- 4 = Good
- 5 = Excellent

---

## Emergency Protocols

### If Technology Fails:
1. **Have printed slides** as backup  
2. **Explain from memory** - you know this content cold  
3. **Draw on whiteboard** if needed (especially weight formula, algorithm)  
4. **Stay calm** - committee wants to hear your ideas, not watch PowerPoint

### If You Blank on a Slide:
1. **Glance at speaker notes** (that's why they exist)  
2. **Describe the figure** - visual cues trigger memory  
3. **Bridge to next slide** - "Let me move to the next point which builds on this..."  
4. **Don't apologize excessively** - brief "excuse me" then continue

### If You Run Out of Time:
1. **Skip to conclusions** (slide 19) at 19-minute mark  
2. **Summarize remaining slides** in 30 seconds: "The data shows limitations on COMPAS, practical guidance favors decision-based applications, and future work includes calibration restoration."  
3. **End gracefully** - better to finish on time than rush through

### If You Finish Too Early (< 17 minutes):
1. **Expand on mechanism** (slide 11) - this is your most interesting finding  
2. **Show additional results** from thesis if asked  
3. **Invite questions early** - "I have time for questions if the committee has any before we move to Q&A"

---

## Day 29 Goals

### Minimum Success Criteria:
✅ Complete 3 full practice runs (solo, timing varies)  
✅ Memorize opening and closing statements word-for-word  
✅ Practice Q&A answers for all 8 anticipated questions  
✅ Achieve timing: 18-22 minutes consistently

### Stretch Goals:
✅ Record practice run and review video  
✅ Practice in front of friend/family for feedback  
✅ Create backup slides for technical questions  
✅ Prepare answers to 5 additional hypothetical questions

---

## Final Mindset

### Remember:
1. **You are the expert** on this thesis - you know more than anyone in the room about your specific method  
2. **Perfect fairness (EO=0.0) is a big deal** - be proud of this achievement  
3. **It's okay not to know everything** - "I don't know, but here's how I'd investigate..." is a valid answer  
4. **The committee wants you to succeed** - they're evaluating, not attacking  
5. **You've done excellent work** - first-class BSc thesis quality

### Confidence Builders:
- 71-page professional thesis ✅  
- Perfect fairness on real data ✅  
- Novel weighting formula ✅  
- First calibration-fairness quantification ✅  
- 27 days of rigorous experimentation ✅  
- Publication-worthy findings ✅

**You've got this!** 🎓🚀

---

## Post-Practice Reflection

After your final practice run, answer these:

1. What am I most confident about?
   - _______________________________

2. What needs more practice?
   - _______________________________

3. Which Q&A answer needs refinement?
   - _______________________________

4. What's my backup plan if I blank?
   - _______________________________

5. Why am I proud of this work?
   - _______________________________

---

**Practice Schedule for Day 29:**
- **Morning (9-10 AM):** Session 1 - Full run-through  
- **Midday (1-2 PM):** Session 2 - Slow practice  
- **Afternoon (3-4 PM):** Session 3 - Timed run  
- **Evening (7-8 PM):** Session 4 - Q&A practice

**Goal:** By end of Day 29, you should be able to deliver the presentation smoothly with 90% confidence.

**Next:** Day 30 - Final polish, equipment check, mental preparation.
