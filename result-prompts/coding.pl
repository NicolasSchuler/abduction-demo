% =============================================================================
% ==  1. CORE CAUSAL MODEL (LOGIC)
% =============================================================================
% This section defines the fundamental logic: animal type causes features.

% -- Priors: Before any evidence, we assume a 50/50 chance.
0.5::is_cat; 0.5::is_dog.

% -- General Rule: Animal type causes its physical features.
P::feature(F) :- is_cat, prob_cat(F, P).
P::feature(F) :- is_dog, prob_dog(F, P).


% =============================================================================
% ==  2. KNOWLEDGE BASE (DATA: P(Feature | Animal))
% =============================================================================
% This is the program's "encyclopedia" of facts about cats and dogs.
% Atoms are based on reasoning.md, probabilities are from the original file.

% -- Skull Morphology and Facial Structure
prob_cat(flat_forehead, 0.9).            prob_dog(flat_forehead, 0.1).
prob_cat(small_muzzle, 0.95).             prob_dog(small_muzzle, 0.2).
prob_cat(short_rounded_face, 0.8).        prob_dog(short_rounded_face, 0.2).
prob_cat(longer_snout, 0.05).             prob_dog(longer_snout, 0.9).
prob_cat(pronounced_nasal_bridge, 0.1).   prob_dog(pronounced_nasal_bridge, 0.8).
prob_cat(elongated_skull, 0.2).           prob_dog(elongated_skull, 0.7).

% -- Nose and Nasal Structure
prob_cat(small_upturned_nose, 0.9).        prob_dog(small_upturned_nose, 0.2).
prob_cat(large_robust_nose, 0.1).          prob_dog(large_robust_nose, 0.8).
prob_cat(visible_nasal_bridge_nose, 0.01). prob_dog(visible_nasal_bridge_nose, 0.4).

% -- Ear Shape and Position
prob_cat(small_pointed_upright_ears, 0.95). prob_dog(small_pointed_upright_ears, 0.2).
prob_cat(high_placement_ears, 0.9).         prob_dog(high_placement_ears, 0.3).
prob_cat(vertically_oriented_ears, 0.95).   prob_dog(vertically_oriented_ears, 0.4).
prob_cat(mobile_ears, 0.9).                 prob_dog(mobile_ears, 0.8).
prob_cat(variable_ear_shape, 0.1).          prob_dog(variable_ear_shape, 0.9).
prob_cat(large_drooping_or_folded_ears, 0.05). prob_dog(large_drooping_or_folded_ears, 0.8).

% -- Eye Position and Size
prob_cat(vertical_pupils, 0.95).            prob_dog(vertical_pupils, 0.1).
prob_cat(close_set_forward_eyes, 0.9).      prob_dog(close_set_forward_eyes, 0.5).
prob_cat(large_forward_facing_eyes, 0.9).   prob_dog(large_forward_facing_eyes, 0.2).
prob_cat(round_pupils, 0.05).               prob_dog(round_pupils, 0.9).
prob_cat(lateral_eye_placement, 0.1).       prob_dog(lateral_eye_placement, 0.5).
prob_cat(variable_eye_size, 0.01).          prob_dog(variable_eye_size, 0.3).

% -- Tail Morphology
prob_cat(upright_tail, 0.9).                prob_dog(upright_tail, 0.2).
prob_cat(long_slender_tail, 0.8).           prob_dog(long_slender_tail, 0.5).
prob_cat(robust_variable_tail, 0.2).        prob_dog(robust_variable_tail, 0.8).
prob_cat(curled_tail, 0.05).                prob_dog(curled_tail, 0.7).

% -- Body Proportions and Posture
prob_cat(flexible_spine, 0.95).             prob_dog(flexible_spine, 0.2).
prob_cat(crouched_posture, 0.8).             prob_dog(crouched_posture, 0.4).
prob_cat(arched_back, 0.7).                 prob_dog(arched_back, 0.4).
prob_cat(rigid_spine, 0.05).                prob_dog(rigid_spine, 0.8).
prob_cat(elongated_body, 0.1).               prob_dog(elongated_body, 0.7).
prob_cat(straight_back, 0.3).               prob_dog(straight_back, 0.6).

% -- Paw Structure and Claws
prob_cat(small_defined_paw_pads, 0.9).      prob_dog(small_defined_paw_pads, 0.1).
prob_cat(small_pad_like_feet, 0.8).         prob_dog(small_pad_like_feet, 0.4).
prob_cat(retractable_claws, 0.99).          prob_dog(retractable_claws, 0.01).
prob_cat(large_robust_paw_pads, 0.1).       prob_dog(large_robust_paw_pads, 0.9).
prob_cat(large_feet, 0.2).                  prob_dog(large_feet, 0.6).
prob_cat(non_retractable_claws, 0.01).      prob_dog(non_retractable_claws, 0.99).

% -- Fur Texture and Density
prob_cat(smooth_dense_fur, 0.8).            prob_dog(smooth_dense_fur, 0.9).
prob_cat(variable_coarse_fur, 0.2).         prob_dog(variable_coarse_fur, 0.6).


% =============================================================================
% ==  3. OBSERVATION MODEL (PER-OBSERVATION)
% =============================================================================
% This models the link between a real feature and an observer's report about it.

% P(report | feature is true) is the True Positive Rate (TPR) for that observation.
TPR::report(Observer, Feature) :- feature(Feature), confidence(Observer, Feature, TPR, _FPR).

% P(report | feature is false) is the False Positive Rate (FPR) for that observation.
FPR::report(Observer, Feature) :- \+feature(Feature), confidence(Observer, Feature, _TPR, FPR).


% =============================================================================
% ==  4. YOUR EVIDENCE (EDIT THIS SECTION)
% =============================================================================
% For each observation, provide a 'confidence' fact and an 'evidence' fact.
% --- Atoms here are updated to match Section 2 ---
% This section will be populated by the execute_logic_program function

% --- Example Scenario: Two people observe an animal that looks like a cat ---

% ## Observer 1: Very confident ##
% "I'm 95% certain I saw a small muzzle."
% Rule: For X% confidence on a POSITIVE sighting, use (X/100) for TPR and ((100-X)/100) for FPR.
% confidence(observer1, small_muzzle, 0.95, 0.05).
% evidence(report(observer1, small_muzzle), true).

% "I'm 99% certain I saw retractable claws."
% confidence(observer1, retractable_claws, 0.99, 0.01).
% evidence(report(observer1, retractable_claws), true).


% ## Observer 2: Less confident ##
% "I think I saw small, pointed, upright ears, but I'm only 70% sure."
% confidence(observer2, small_pointed_upright_ears, 0.70, 0.30).
% evidence(report(observer2, small_pointed_upright_ears), true).

% "I'm 90% sure it did NOT have large, drooping, or folded ears."
% Rule: For Y% confidence on a NEGATIVE sighting, use ((100-Y)/100) for both TPR and FPR.
% confidence(observer2, large_drooping_or_folded_ears, 0.10, 0.10).
% evidence(report(observer2, large_drooping_or_folded_ears), false).


% =============================================================================
% ==  5. QUERY
% =============================================================================
% Asks the program for the final probability of each hypothesis.
% These queries will be added by the execute_logic_program function
% query(is_cat).
% query(is_dog).