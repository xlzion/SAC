# Top-4 Security Conference Papers Related to SAC

Date: 2026-06-05

Scope: IEEE S&P, USENIX Security, ACM CCS, and NDSS papers most useful for a paper on Security-Aware Selective Compression (SAC) for mitigating LoRA/PEFT LLM backdoors.

## Highest-priority citations

1. **PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**  
   Venue: IEEE S&P 2025  
   Why it fits: Directly targets PEFT/LoRA adapter backdoors and shared adapter risk. It is the closest top-security-venue paper for positioning SAC as mitigation rather than detection.  
   Suggested placement: Related Work, "LoRA adaptation and PEFT security"; Introduction threat framing.  
   Source: https://www.ieee-security.org/TC/SP2025/program.html and https://arxiv.org/abs/2411.17453

2. **The Philosopher's Stone: Trojaning Plugins of Large Language Models**  
   Venue: NDSS 2026  
   Why it fits: Direct LoRA-adapter trojan paper. It shows infected adapters can trigger adversary-chosen behavior and even malicious tool use. Strong support for the adapter supply-chain threat model.  
   Suggested placement: Related Work, "LLM backdoors and sleeper behavior"; Introduction threat framing.  
   Source: https://www.ndss-symposium.org/ndss-paper/the-philosophers-stone-trojaning-plugins-of-large-language-models/

3. **Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models**  
   Venue: NDSS 2026  
   Why it fits: Direct open-weight LoRA backdoor work, with a causal-guided weight-allocation mechanism and post-training control over attack intensity. Useful contrast to SAC's causal/counterfactual component scoring.  
   Suggested placement: Related Work, "LLM backdoors and sleeper behavior"; Discussion or limitations if talking about adaptive attacks.  
   Source: https://www.ndss-symposium.org/ndss-paper/causal-guided-detoxify-backdoor-attack-of-open-weight-lora-models/

4. **BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target**  
   Venue: IEEE S&P 2025  
   Why it fits: LLM-specific backdoor scanning for generative outputs. SAC is not a scanner, but BAIT is a strong top-security reference for why LLM backdoor detection differs from discriminative-model backdoor detection.  
   Suggested placement: Related Work, "Backdoor defenses and safety tuning"; Motivation for post-hoc adapter-only mitigation.  
   Source: https://www.ieee-security.org/TC/SP2025/program.html and https://www.cs.purdue.edu/homes/shen447/files/paper/sp25_bait.pdf

5. **Instruction Backdoor Attacks Against Customized LLMs**  
   Venue: USENIX Security 2024  
   Why it fits: Shows customized LLM artifacts can embed backdoor behavior without modifying the backend model. This supports the general "trusted base, untrusted customization artifact" framing behind malicious adapters.  
   Suggested placement: Introduction threat framing; Related Work, "LLM backdoors and sleeper behavior."  
   Source: https://www.usenix.org/conference/usenixsecurity24/presentation/zhang-rui

6. **EmbedX: Embedding-Based Cross-Trigger Backdoor Attack Against Large Language Models**  
   Venue: USENIX Security 2025  
   Why it fits: LLM backdoor attack with soft/cross-trigger design. Useful for arguing that trigger-conditioned behavior can generalize beyond single literal triggers.  
   Suggested placement: Related Work, "LLM backdoors and sleeper behavior"; Evaluation caveats on trigger diversity.  
   Source: https://www.usenix.org/conference/usenixsecurity25/presentation/yan-nan

7. **Mudjacking: Patching Backdoor Vulnerabilities in Foundation Models**  
   Venue: USENIX Security 2024  
   Why it fits: Post-deployment backdoor removal for foundation models. SAC can be positioned as an adapter-only, compression-budgeted post-hoc mitigation, complementary to full foundation-model patching.  
   Suggested placement: Related Work, "Backdoor defenses and safety tuning."  
   Source: https://www.usenix.org/conference/usenixsecurity24/presentation/liu-hongbin

8. **JBShield: Defending Large Language Models from Jailbreak Attacks through Activated Concept Analysis and Manipulation**  
   Venue: USENIX Security 2025  
   Why it fits: Uses concept/representation analysis to detect and mitigate jailbreaks. Good comparison point for SAC's behavior-grounded component scoring and four-way refusal/ASR evaluation.  
   Suggested placement: Related Work, "Backdoor defenses and safety tuning"; "Editing and causal localization."  
   Source: https://www.usenix.org/conference/usenixsecurity25/presentation/zhang-shenyi

## Useful secondary citations

9. **Hidden Trigger Backdoor Attack on NLP Models via Linguistic Style Manipulation**  
   Venue: USENIX Security 2022  
   Why it fits: Classic top-security NLP hidden-trigger paper. Good background for stealthy textual triggers and why word-level trigger defenses are insufficient.  
   Source: https://www.usenix.org/conference/usenixsecurity22/presentation/pan-hidden

10. **Neural Network Semantic Backdoor Detection and Mitigation: A Causality-Based Approach**  
    Venue: USENIX Security 2024  
    Why it fits: Causality-based semantic backdoor detection/mitigation. Useful background for causality/counterfactual component contribution, though not LLM/LoRA-specific.  
    Source: https://www.usenix.org/conference/usenixsecurity24/presentation/sun-bing

11. **BadMerging: Backdoor Attacks Against Model Merging**  
    Venue: ACM CCS 2024  
    Why it fits: Modular model-supply-chain threat where a contributed model can compromise merged behavior. This is adjacent to third-party adapter loading and merging.  
    Source: https://www.sigsac.org/ccs/CCS2024/program/accepted-papers.html and https://arxiv.org/abs/2408.07362

12. **A Causal Explainable Guardrails for Large Language Models**  
    Venue: ACM CCS 2024  
    Why it fits: LLM guardrail work using causal analysis and steering representations. Helpful for positioning SAC relative to representation-level guardrail methods.  
    Source: https://www.sigsac.org/ccs/CCS2024/program/accepted-papers.html and https://arxiv.org/abs/2405.04160

13. **"Do Anything Now": Characterizing and Evaluating In-The-Wild Jailbreak Prompts on Large Language Models**  
    Venue: ACM CCS 2024  
    Why it fits: Empirical jailbreak prompt study. Useful for grounding AdvBench/HarmBench-style external attack evaluation and in-the-wild jailbreak framing.  
    Source: https://www.sigsac.org/ccs/CCS2024/program/accepted-papers.html and https://arxiv.org/abs/2308.03825

14. **Fisher Information guided Purification against Backdoor Attacks**  
    Venue: ACM CCS 2024  
    Why it fits: Backdoor purification defense. Generic model defense rather than LLM/LoRA-specific, but useful as a nearby post-hoc purification baseline family.  
    Source: https://www.sigsac.org/ccs/CCS2024/program/accepted-papers.html and https://arxiv.org/abs/2409.00863

15. **TwinBreak: Jailbreaking LLM Security Alignments based on Twin Prompts**  
    Venue: USENIX Security 2025  
    Why it fits: Opposite-direction pruning work: it identifies and prunes safety-alignment parameters to jailbreak models. This is an important contrast because SAC uses selective compression to remove harmful triggered behavior while preserving refusal and benign utility.  
    Source: https://www.usenix.org/conference/usenixsecurity25/presentation/krauss

16. **Mind the Inconspicuous: Revealing the Hidden Weakness in Aligned LLMs' Refusal Boundaries**  
    Venue: USENIX Security 2025  
    Why it fits: Refusal-boundary analysis for aligned LLMs. Useful for arguing that refusal/over-refusal must be measured separately from ASR, which SAC's TH/H/TB/B protocol does.  
    Source: https://www.usenix.org/conference/usenixsecurity25/presentation/yu-jiahao

17. **Alleviating the Fear of Losing Alignment in LLM Fine-tuning**  
    Venue: IEEE S&P 2025  
    Why it fits: Fine-tuning can degrade alignment; the paper focuses on recovering alignment after fine-tuning. Good adjacent citation for safety degradation and post-fine-tuning repair.  
    Source: https://www.ieee-security.org/TC/SP2025/program.html and https://arxiv.org/abs/2504.09757

18. **Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface**  
    Venue: IEEE S&P 2025  
    Why it fits: Fine-tuning interface as an attack surface for prompt-injection-style behavior. Less direct than PEFTGuard, but useful for a broad "fine-tuning/customization creates security risk" sentence.  
    Source: https://www.ieee-security.org/TC/SP2025/accepted-papers.html

## 2026 papers to watch or cite as concurrent/fresh background

19. **SoK: Robustness in Large Language Models against Jailbreak Attacks**  
    Venue: IEEE S&P 2026  
    Why it fits: Survey/systematization reference for jailbreak robustness. Use if the paper needs an up-to-date top-security SoK citation.  
    Source: https://www.ieee-security.org/TC/SP2026/accepted-papers.html

20. **SoK: Evaluating Jailbreak Guardrails for Large Language Models**  
    Venue: IEEE S&P 2026  
    Why it fits: Guardrail evaluation reference. Good for a sentence about why evaluating only attack success is insufficient.  
    Source: https://www.ieee-security.org/TC/SP2026/accepted-papers.html

21. **URLcoat: Exploiting Web Search Capability to Jailbreak Large Language Models**  
    Venue: IEEE S&P 2026  
    Why it fits: New jailbreak attack through web search capability. Not central to LoRA backdoors, but relevant if expanding external attack suites or deployment threat surfaces.  
    Source: https://www.ieee-security.org/TC/SP2026/accepted-papers.html

22. **PromptLocate: Localizing Prompt Injection Attacks**  
    Venue: IEEE S&P 2026  
    Why it fits: Localization of prompt injection. Not a backdoor-compression paper, but useful if discussing localization-style defenses and causal attribution.  
    Source: https://www.ieee-security.org/TC/SP2026/accepted-papers.html

## Suggested citation strategy for SAC

- Add PEFTGuard, Philosopher's Stone, and Causal-Guided Detoxify Backdoor Attack as direct LoRA/PEFT security anchors.
- Add BAIT, Mudjacking, JBShield, and SODA as post-hoc detection/mitigation/causal-analysis anchors.
- Add BadMerging and Instruction Backdoor Attacks for the modular customization supply-chain framing.
- Add Do Anything Now, Mind the Inconspicuous, and the 2026 SoK papers if the paper needs stronger justification for jailbreak/refusal evaluation.
- Keep compression papers separate from security-top-conference papers: SAC's novelty is the intersection of compression and security-aware adapter intervention.
