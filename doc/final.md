
# Final Report: Full Robot Evolution

**Goal:** Evolve a complete robot — body morphology and neural controller — capable of locomotion across four environments.  
**How:** Integrate the body–brain co-evolution pipeline from the previous challenges and extend it to a multi-task setting.

---

![evorob overview](imgs/evorob_overview.png)

## Introduction

The goal of the final project is to design and conduct an evolutionary experiment in which a legged robot is evolved to walk in the positive x-direction across four environments: flat terrain, ice terrain, hilly terrain, and a mystery terrain that will be disclosed at the start of the poster session.

Before running experiments, formulate a clear research question and corresponding hypothesis. The following axes of investigation are suggested, though students are free to propose alternatives:

- Custom geno2pheno mappings or morphological representations
- Alternative controller architectures (SO2-oscillators, Hebbian networks, MLPs)
- Body evolution combined with lifetime learning or online adaptation
- Effect of terrain diversity on the evolved morphology

**Scope:** Limit the investigation to a single axis of comparison (e.g. controller type). State a hypothesis about the expected differences in morphological design and control strategy across conditions prior to running experiments.

Ask questions for guidance when stuck!

The final grade is assessed in a poster session.

---

## Final Project Submission Details
The final project should demonstrate a thorough understanding of body–brain co-evolution and multi-task optimisation in the context of legged locomotion.

To complete the final project, submit the following materials:

- **Code**: Your `final_project.py`, `<your_controller>.py` with all experimental modifications
- **Data**: Your best evolved genotype as `x_best.npy`

Provide all documents in a zipped folder using the following naming convention:  
`2026_micro_515_SCIPER_TEAMNAME_LASTNAME1_LASTNAME2_final.zip`

---

## Questions?

If some parts of your code are not working or you have general questions, do not hesitate to contact your MICRO-515 teaching assistants in the exercise sessions or via e-mail `fuda.vandiggelen@epfl.ch`, `alexander.ertl@epfl.ch`, `alexander.dittrich@epfl.ch`, `hongze.wang@epfl.ch`