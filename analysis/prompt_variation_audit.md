# Prompt Variation Matching Audit

For each original prompt, show 5 matched variations and whether they truly ask the same question.


## acs_severity (4 original prompts, 403 variations)

### ORIGINAL: "Does this patient have an acute coronary occlusion? If present, specify whether it is complete or incomplete and identify the culprit artery."

  - [1.00] ✅ MATCH: "Does this patient have an acute coronary occlusion? If present, specify whether it is complete or incomplete and identify the culprit artery."
  - [0.81] ✅ MATCH: "Does this patient have acute coronary occlusion? If yes, specify completeness and identify the culprit vessel."
  - [0.77] ✅ MATCH: "Does this patient have an acute coronary occlusion? If yes, describe its completeness and the culprit vessel."
  - [0.76] ✅ MATCH: "Identify whether an acute coronary occlusion is present, whether it's complete or incomplete, and which artery is the culprit."
  - [0.75] ✅ MATCH: "Please evaluate for acute coronary occlusion. Report whether it's complete or incomplete and identify the culprit."
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Does this show partial or complete LAD occlusion?"
  - [0.35] ❌ "What does this ECG reveal about coronary artery patency?"

### ORIGINAL: "Is there evidence of an acute coronary occlusion? Please state the completeness and the likely culprit artery."

  - [1.00] ✅ MATCH: "Is there evidence of an acute coronary occlusion? Please state the completeness and the likely culprit artery."
  - [0.74] ✅ MATCH: "Can you detect an acute coronary occlusion? State its completeness and the involved artery."
  - [0.71] ✅ MATCH: "What is the acute coronary occlusion assessment? Determine completeness and identify the culprit artery."
  - [0.70] ✅ MATCH: "Please interpret this ECG regarding acute coronary occlusion. State completeness and culprit artery."
  - [0.70] ✅ MATCH: "Review the ECG for acute coronary occlusion findings. Report completeness and culprit artery name."
  **WORST MATCHES (potential misassignments):**
  - [0.19] ❌ "What vessels are blocked?"
  - [0.21] ❌ "What coronary artery is responsible for the ischemic changes seen here?"

### ORIGINAL: "Is this an acute coronary occlusion? If yes, indicate whether it is complete or incomplete and name the culprit artery."

  - [1.00] ✅ MATCH: "Is this an acute coronary occlusion? If yes, indicate whether it is complete or incomplete and name the culprit artery."
  - [0.82] ✅ MATCH: "Analyze for evidence of acute coronary occlusion - indicate whether complete or incomplete and name the culprit artery."
  - [0.81] ✅ MATCH: "Please report on any acute coronary occlusion, stating whether it's complete or incomplete and the culprit artery."
  - [0.81] ✅ MATCH: "Please assess this ECG for acute coronary occlusion. Indicate whether complete or incomplete and name the culprit."
  - [0.81] ✅ MATCH: "Inspect for acute coronary occlusion. State whether it's complete or incomplete and identify the culprit artery."
  **WORST MATCHES (potential misassignments):**
  - [0.29] ❌ "Can you identify the culprit vessel responsible for these ECG changes?"
  - [0.31] ❌ "This ECG shows which artery blockage?"

### ORIGINAL: "IS there an acute coornary occlusion/ if yes is it ocmplete or incompelte and what is the culprit?"

  - [0.98] ✅ MATCH: "IS there an acute coornary occlusion/ if yes is it complete or incomplete and what is the culprit?"
  - [0.84] ✅ MATCH: "Is there an acute coronary occlusion? If so, is it complete or partial, and which artery is the culprit?"
  - [0.72] ✅ MATCH: "Determine the presence of acute coronary occlusion. Specify if complete or incomplete and name the culprit."
  - [0.66] ✅ MATCH: "What does the ECG reveal about coronary occlusion? Is it complete or incomplete, and which vessel is involved?"
  - [0.65] ✅ MATCH: "Is there an acute coronary occlusion present? If affirmative, describe its nature and identify the culprit."
  **WORST MATCHES (potential misassignments):**
  - [0.16] ❌ "What arteries are involved?"
  - [0.19] ❌ "Which vessel demonstrates obstruction?"


## afib_risk (5 original prompts, 400 variations)

### ORIGINAL: "Will this patient develop atrial fibrillation in the future?"

  - [1.00] ✅ MATCH: "Will this patient develop atrial fibrillation in the future?"
  - [0.87] ✅ MATCH: "Will this patient develop atrial fibrillation in the coming years?"
  - [0.78] ✅ MATCH: "Is this patient likely to develop atrial fibrillation in the coming years?"
  - [0.72] ✅ MATCH: "Could this patient potentially develop atrial fibrillation?"
  - [0.72] ✅ MATCH: "Could we expect this patient to develop atrial fibrillation?"

### ORIGINAL: "What is the patient's risk of developing atrial fibrillation?"

  - [1.00] ✅ MATCH: "What is the patient's risk of developing atrial fibrillation?"
  - [0.91] ✅ MATCH: "What is this patient's lifetime risk of developing atrial fibrillation?"
  - [0.90] ✅ MATCH: "What is the patient's outlook for developing atrial fibrillation?"
  - [0.90] ✅ MATCH: "What is the patient's projected risk for developing atrial fibrillation?"
  - [0.90] ✅ MATCH: "What is the patient's estimated risk for developing atrial fibrillation?"

### ORIGINAL: "What is the likelihood of developing AFib in the next 2-5 years?"

  - [1.00] ✅ MATCH: "What is the likelihood of developing AFib in the next 2-5 years?"
  - [0.69] ✅ MATCH: "What are the chances of AFib developing within the next several years?"
  - [0.63] ✅ MATCH: "What is the estimated likelihood of AFib in this patient?"
  - [0.62] ✅ MATCH: "Estimate the likelihood of AF onset within the next decade."
  - [0.62] ✅ MATCH: "What is the anticipated growth in AFib risk over the coming years?"

### ORIGINAL: "Is this patient at risk for incident AFib?"

  - [1.00] ✅ MATCH: "Is this patient at risk for incident AFib?"
  - [0.90] ✅ MATCH: "Is this patient at elevated risk for incident AFib?"
  - [0.77] ✅ MATCH: "Is this patient at increased risk for developing AFib?"
  - [0.72] ✅ MATCH: "Would you consider this patient high-risk for incident AF?"
  - [0.71] ✅ MATCH: "Evaluate the patient's risk factors for incident AFib."

### ORIGINAL: "What is the patient's future AFib risk?"

  - [1.00] ✅ MATCH: "What is the patient's future AFib risk?"
  - [0.75] ✅ MATCH: "What is the patient's projected AFib development risk?"
  - [0.74] ✅ MATCH: "How would you assess this patient's future AFib risk?"
  - [0.70] ✅ MATCH: "Evaluate the patient's future atrial fibrillation risk."
  - [0.69] ✅ MATCH: "How significant is the patient's exposure to AFib risk factors?"


## category_chamber_enlargement (10 original prompts, 600 variations)

### ORIGINAL: "Does this show ventricular hypertrophy?"

  - [1.00] ✅ MATCH: "Does this show ventricular hypertrophy?"
  - [0.78] ✅ MATCH: "Do the precordial voltages show ventricular hypertrophy?"
  - [0.75] ✅ MATCH: "Does this ECG show criteria meeting left ventricular hypertrophy?"
  - [0.74] ✅ MATCH: "Does this show the electrical signature of hypertrophy?"
  - [0.74] ✅ MATCH: "Does the axis suggest ventricular hypertrophy or dilation?"
  **WORST MATCHES (potential misassignments):**
  - [0.35] ❌ "What structural information does this ECG provide about chambers?"
  - [0.35] ❌ "What cardiac structural information does this ECG provide?"

### ORIGINAL: "Can you assess for chamber abnormalities?"

  - [1.00] ✅ MATCH: "Can you assess for chamber abnormalities?"
  - [0.70] ✅ MATCH: "Can you find any chamber abnormality indicators?"
  - [0.69] ✅ MATCH: "Can you assess if chamber sizes appear normal or enlarged?"
  - [0.67] ✅ MATCH: "Please evaluate for any chamber abnormalities."
  - [0.63] ✅ MATCH: "Please analyze the QRS voltages for chamber abnormalities."

### ORIGINAL: "Are there signs of LVH or RVH?"

  - [1.00] ✅ MATCH: "Are there signs of LVH or RVH?"
  - [0.54] ✅ MATCH: "Assess the ECG for signs of pulmonary heart disease."
  - [0.53] ✅ MATCH: "Are there any ECG markers of pulmonary heart disease?"
  - [0.52] ✅ MATCH: "Are there any indicators of cardiac chamber change?"
  - [0.45] ✅ MATCH: "Check for the voltage criteria of Sokolow-Lyon for LVH."

### ORIGINAL: "Is there evidence of atrial abnormality?"

  - [1.00] ✅ MATCH: "Is there evidence of atrial abnormality?"
  - [0.84] ✅ MATCH: "Is there evidence suggesting left atrial abnormality?"
  - [0.83] ✅ MATCH: "Is there diagnostic evidence of left atrial abnormality?"
  - [0.78] ✅ MATCH: "Does the P wave show evidence of right atrial abnormality?"
  - [0.76] ✅ MATCH: "Evaluate the P wave for evidence of atrial abnormality."

### ORIGINAL: "Is there chamber enlargement or hypertrophy?"

  - [1.00] ✅ MATCH: "Is there chamber enlargement or hypertrophy?"
  - [0.74] ✅ MATCH: "Is there evidence of chamber dilation or hypertrophy?"
  - [0.62] ✅ MATCH: "Is there voltage evidence of chamber hypertrophy?"
  - [0.62] ✅ MATCH: "Is there radiographic evidence of chamber enlargement in the tracing?"
  - [0.61] ✅ MATCH: "Analyze the ECG for signs of chamber dilation or hypertrophy."

### ORIGINAL: "Is there biatrial or biventricular enlargement?"

  - [1.00] ✅ MATCH: "Is there biatrial or biventricular enlargement?"
  - [0.74] ✅ MATCH: "Does the ECG show evidence of atrial or ventricular enlargement?"
  - [0.73] ✅ MATCH: "Are there criteria suggesting left or right ventricular enlargement?"
  - [0.72] ✅ MATCH: "Is there any suggestion of ventricular or atrial enlargement?"
  - [0.72] ✅ MATCH: "Please examine whether atrial or ventricular enlargement exists."
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "What hypertrophic patterns emerge from this ECG analysis?"
  - [0.36] ❌ "What cardiac structural indicators appear in this ECG?"

### ORIGINAL: "Are there signs of ventricular strain?"

  - [1.00] ✅ MATCH: "Are there signs of ventricular strain?"
  - [0.74] ✅ MATCH: "Are there any signs of right ventricular pathology?"
  - [0.69] ✅ MATCH: "Is there any sign of right ventricular strain or hypertrophy?"
  - [0.65] ✅ MATCH: "Are there any signs of right heart strain or hypertrophy?"
  - [0.62] ✅ MATCH: "Please examine for evidence of ventricular strain pattern."
  **WORST MATCHES (potential misassignments):**
  - [0.35] ❌ "What cardiac size information appears in this tracing?"
  - [0.37] ❌ "What hypertrophic patterns are present in the tracing?"

### ORIGINAL: "What do the voltages suggest?"

  - [1.00] ✅ MATCH: "What do the voltages suggest?"
  - [0.68] ✅ MATCH: "What does the voltage pattern suggest about chambers?"
  - [0.66] ✅ MATCH: "What does the axis and voltages suggest about chamber size?"
  - [0.65] ✅ MATCH: "What does the QRS voltage suggest about ventricular size?"
  - [0.63] ✅ MATCH: "What chamber abnormalities does the voltage pattern suggest?"
  **WORST MATCHES (potential misassignments):**
  - [0.30] ❌ "What cardiac size information can be interpreted from these waveforms?"
  - [0.35] ❌ "What hypertrophic indicators are present in the ECG?"

### ORIGINAL: "Is there atrial enlargement?"

  - [1.00] ✅ MATCH: "Is there atrial enlargement?"
  - [0.75] ✅ MATCH: "Does this show biatrial enlargement?"
  - [0.67] ✅ MATCH: "Is there indication of left or right atrial enlargement?"
  - [0.67] ✅ MATCH: "Is there any indication of chamber enlargement?"
  - [0.66] ✅ MATCH: "Does this tracing suggest right atrial enlargement?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Describe any morphological features indicating abnormal chamber dimensions."
  - [0.36] ❌ "What structural heart findings emerge from this ECG analysis?"

### ORIGINAL: "Are the QRS voltages normal?"

  - [1.00] ✅ MATCH: "Are the QRS voltages normal?"
  - [0.64] ✅ MATCH: "Are the QRS voltages within normal limits for chamber size?"
  - [0.62] ✅ MATCH: "Are the QRS voltages consistent with normal chamber dimensions?"
  - [0.61] ✅ MATCH: "Can the QRS voltages indicate chamber problems?"
  - [0.61] ✅ MATCH: "Assess whether the QRS voltages exceed normal thresholds."


## category_conduction (10 original prompts, 600 variations)

### ORIGINAL: "What type of block is present if any?"

  - [1.00] ✅ MATCH: "What type of block is present if any?"
  - [0.72] ✅ MATCH: "What type of block, if any, is present in this tracing?"
  - [0.71] ✅ MATCH: "What type of cardiac block, if any, is present?"
  - [0.71] ✅ MATCH: "What is the degree of block if present?"
  - [0.69] ✅ MATCH: "What block type is present?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Please evaluate the transmission of impulses through the AV node."
  - [0.35] ❌ "What abnormalities exist in impulse transmission through the heart?"

### ORIGINAL: "Is there any conduction delay?"

  - [1.00] ✅ MATCH: "Is there any conduction delay?"
  - [0.80] ✅ MATCH: "Is there evidence of a conduction delay?"
  - [0.80] ✅ MATCH: "Are there any conduction delays to note?"
  - [0.78] ✅ MATCH: "Is there a conduction disturbance?"
  - [0.77] ✅ MATCH: "Check for any conduction delays."
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Look for any abnormal findings in the atrioventricular transmission pathway"
  - [0.34] ❌ "Please examine the pathway of electrical activation through the myocardium."

### ORIGINAL: "Are there any bundle branch blocks?"

  - [1.00] ✅ MATCH: "Are there any bundle branch blocks?"
  - [0.82] ✅ MATCH: "Is there a left bundle branch block?"
  - [0.76] ✅ MATCH: "Is there a left bundle branch block present?"
  - [0.72] ✅ MATCH: "Assess for the presence of bundle branch blocks."
  - [0.68] ✅ MATCH: "Check for any incomplete bundle branch block."
  **WORST MATCHES (potential misassignments):**
  - [0.30] ❌ "Please evaluate the electrical sequence from SA node to the ventricles."
  - [0.33] ❌ "Evaluate whether the electrical signal is being transmitted properly"

### ORIGINAL: "Can you assess the conduction system?"

  - [1.00] ✅ MATCH: "Can you assess the conduction system?"
  - [0.78] ✅ MATCH: "Can you detect conduction system issues?"
  - [0.74] ✅ MATCH: "Could you assess the septal conduction?"
  - [0.71] ✅ MATCH: "Can you identify conduction system pathology?"
  - [0.70] ✅ MATCH: "What is the conduction system status?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Analyze the electrical pathway from sinoatrial node to ventricles"
  - [0.34] ❌ "Assess whether the electrical signal reaches the ventricles appropriately."

### ORIGINAL: "Are there any conduction abnormalities?"

  - [1.00] ✅ MATCH: "Are there any conduction abnormalities?"
  - [0.88] ✅ MATCH: "Is there any conduction abnormality?"
  - [0.85] ✅ MATCH: "Are there conduction abnormalities present?"
  - [0.82] ✅ MATCH: "Are there any subtle conduction abnormalities to report?"
  - [0.79] ✅ MATCH: "Examine the AV node for any conduction abnormalities"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Does this tracing show normal impulse propagation through the heart?"
  - [0.35] ❌ "Describe the electrical signal propagation from the atria to the ventricles."

### ORIGINAL: "Is AV conduction normal?"

  - [1.00] ✅ MATCH: "Is AV conduction normal?"
  - [0.73] ✅ MATCH: "Is the bundle branch conduction normal?"
  - [0.63] ✅ MATCH: "Look at the conduction intervals."
  - [0.57] ✅ MATCH: "What findings suggest a conduction problem?"
  - [0.54] ✅ MATCH: "Is the AV node transmitting impulses normally?"

### ORIGINAL: "Is there any heart block present?"

  - [1.00] ✅ MATCH: "Is there any heart block present?"
  - [0.79] ✅ MATCH: "Is there a bifascicular block present?"
  - [0.71] ✅ MATCH: "Is there evidence of any heart block?"
  - [0.68] ✅ MATCH: "Assess the severity of any AV block present."
  - [0.65] ✅ MATCH: "Is there evidence of a first-degree AV block present?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Identify any delays in the electrical pathway from sinoatrial to atrioventricular node"
  - [0.36] ❌ "Is the impulse from the atria reaching the ventricles appropriately?"

### ORIGINAL: "Is there AV dissociation?"

  - [1.00] ✅ MATCH: "Is there AV dissociation?"
  - [0.86] ✅ MATCH: "Is there AV dissociation present?"
  - [0.61] ✅ MATCH: "Look for evidence of AV dissociation."
  - [0.57] ✅ MATCH: "Is there any delay in signal transmission?"
  - [0.55] ✅ MATCH: "Is there a problem with impulse propagation?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Look for any delays in the electrical impulse traveling from atria to ventricles"
  - [0.37] ❌ "Look for any delays in electrical propagation."

### ORIGINAL: "Are there any fascicular blocks?"

  - [1.00] ✅ MATCH: "Are there any fascicular blocks?"
  - [0.74] ✅ MATCH: "Determine if any fascicular blocks exist."
  - [0.72] ✅ MATCH: "Please identify any fascicular block."
  - [0.69] ✅ MATCH: "Is there evidence of a trifascicular block?"
  - [0.69] ✅ MATCH: "Are there any signs of fascicular involvement?"
  **WORST MATCHES (potential misassignments):**
  - [0.31] ❌ "Identify any impaired electrical pathways in this tracing"
  - [0.34] ❌ "Does the tracing show normal sequential ventricular depolarization?"

### ORIGINAL: "Is there evidence of pre-excitation?"

  - [1.00] ✅ MATCH: "Is there evidence of pre-excitation?"
  - [0.85] ✅ MATCH: "Is there any evidence of pre-excitation syndrome?"
  - [0.82] ✅ MATCH: "Is there any evidence of ventricular pre-excitation?"
  - [0.75] ✅ MATCH: "Is there evidence of impaired electrical conduction?"
  - [0.69] ✅ MATCH: "Is there evidence of slowed ventricular conduction?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Report on the status of impulse propagation through the heart."
  - [0.36] ❌ "Please examine the electrical pathway integrity."


## category_infarct_ischemia (9 original prompts, 407 variations)

### ORIGINAL: "Is there any ST elevation or depression?"

  - [1.00] ✅ MATCH: "Is there any ST elevation or depression?"
  - [0.67] ✅ MATCH: "Evaluate the ST segments for elevation or depression."
  - [0.67] ✅ MATCH: "Please analyze the ST segments for elevation or depression."
  - [0.64] ✅ MATCH: "Please analyze the ST segments for any elevation or depression."
  - [0.64] ✅ MATCH: "Please evaluate the ST segments for elevation or depression."

### ORIGINAL: "Are the T waves normal?"

  - [1.00] ✅ MATCH: "Are the T waves normal?"
  - [0.57] ✅ MATCH: "Are the ST segments elevated, depressed, or normal?"
  - [0.51] ✅ MATCH: "Are there T wave changes suggestive of myocardial ischemia?"
  - [0.51] ✅ MATCH: "Are the T wave changes primary or secondary to ischemia?"
  - [0.49] ✅ MATCH: "Examine the ST-T changes across all leads."
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Describe any T wave inversion patterns and their clinical significance."
  - [0.37] ❌ "Please document any T wave inversions and their clinical significance."

### ORIGINAL: "Can you identify any signs of myocardial injury?"

  - [1.00] ✅ MATCH: "Can you identify any signs of myocardial injury?"
  - [0.78] ✅ MATCH: "Identify any signs of myocardial injury in this ECG."
  - [0.67] ✅ MATCH: "Can you detect any subtle evidence of myocardial damage?"
  - [0.65] ✅ MATCH: "Can you identify Q waves that suggest prior myocardial infarction?"
  - [0.63] ✅ MATCH: "What ST segment changes suggest myocardial injury?"
  **WORST MATCHES (potential misassignments):**
  - [0.27] ❌ "Identify which coronary territory appears to be affected based on the pattern."
  - [0.32] ❌ "What pattern of ST changes is present across the precordial leads?"

### ORIGINAL: "Are there any ischemic changes?"

  - [1.00] ✅ MATCH: "Are there any ischemic changes?"
  - [0.70] ✅ MATCH: "Are there any ischemic changes that need urgent attention?"
  - [0.63] ✅ MATCH: "Are there inverted T waves, and do they represent ischemic changes?"
  - [0.60] ✅ MATCH: "Please report any ischemic changes seen in this tracing."
  - [0.59] ✅ MATCH: "Describe any ischemic changes present in this ECG."
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "What myocardial territory shows changes consistent with acute coronary syndrome?"
  - [0.37] ❌ "Are there ST changes that would support an acute coronary syndrome diagnosis?"

### ORIGINAL: "Are there signs of ischemia or infarction?"

  - [1.00] ✅ MATCH: "Are there signs of ischemia or infarction?"
  - [0.74] ✅ MATCH: "Check the ECG for signs of myocardial ischemia or infarction."
  - [0.74] ✅ MATCH: "Check for signs of myocardial ischemia or infarction."
  - [0.65] ✅ MATCH: "Does this tracing show evidence of myocardial ischemia or infarction?"
  - [0.65] ✅ MATCH: "Are there any Q waves that suggest a prior infarction?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "What coronary territory demonstrates changes on this 12-lead tracing?"
  - [0.35] ❌ "What pattern of ST-T abnormalities is demonstrated in this tracing?"

### ORIGINAL: "What do the ST segments show?"

  - [1.00] ✅ MATCH: "What do the ST segments show?"
  - [0.62] ✅ MATCH: "What abnormalities in the ST segments suggest ischemia?"
  - [0.62] ✅ MATCH: "What do the ST segments show in terms of deviation from baseline?"
  - [0.60] ✅ MATCH: "What abnormalities in the ST-T segments suggest ischemia?"
  - [0.57] ✅ MATCH: "What is the significance of the ST segment deviations seen?"
  **WORST MATCHES (potential misassignments):**
  - [0.31] ❌ "What leads would be involved if this represented a left anterior descending artery occlusion?"
  - [0.37] ❌ "What leads would you expect to see changes in with lateral ischemia?"

### ORIGINAL: "Are there pathological Q waves?"

  - [1.00] ✅ MATCH: "Are there pathological Q waves?"
  - [0.72] ✅ MATCH: "Are the QRS complexes showing any pathological Q waves?"
  - [0.68] ✅ MATCH: "Please evaluate for any pathological Q waves."
  - [0.65] ✅ MATCH: "Please assess whether pathological Q waves are present."
  - [0.64] ✅ MATCH: "Does this tracing show evidence of pathological Q waves?"

### ORIGINAL: "Does this ECG show acute MI?"

  - [1.00] ✅ MATCH: "Does this ECG show acute MI?"
  - [0.67] ✅ MATCH: "Does this ECG show Q waves consistent with an old MI?"
  - [0.61] ✅ MATCH: "Does this ECG show signs of myocardial necrosis?"
  - [0.59] ✅ MATCH: "Does this tracing show evidence of myocardial injury?"
  - [0.59] ✅ MATCH: "Does this tracing show evidence of acute coronary occlusion?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Look for any subtle changes that might indicate evolving acute coronary syndrome."
  - [0.36] ❌ "What territorial changes suggest which coronary artery might be involved?"

### ORIGINAL: "Is there evidence of old or new infarction?"

  - [1.00] ✅ MATCH: "Is there evidence of old or new infarction?"
  - [0.83] ✅ MATCH: "Is there evidence of old versus acute infarction?"
  - [0.71] ✅ MATCH: "Can you see evidence of posterior wall infarction?"
  - [0.70] ✅ MATCH: "Is there any evidence of infarction, old or new?"
  - [0.70] ✅ MATCH: "Assess the ECG for evidence of posterior wall infarction."
  **WORST MATCHES (potential misassignments):**
  - [0.35] ❌ "Check for reciprocal ST changes that support infarction diagnosis."
  - [0.38] ❌ "Check for the development of new Q waves compared to any prior tracings."


## category_other (5 original prompts, 400 variations)

### ORIGINAL: "Is there anything else abnormal?"

  - [1.00] ✅ MATCH: "Is there anything else abnormal?"
  - [0.86] ✅ MATCH: "Is there anything else abnormal to report?"
  - [0.71] ✅ MATCH: "Is there anything else abnormal in this electrocardiogram?"
  - [0.70] ✅ MATCH: "Is there anything else that appears abnormal in this record?"
  - [0.65] ✅ MATCH: "Is there evidence of any other abnormalities?"
  **WORST MATCHES (potential misassignments):**
  - [0.27] ❌ "What else can be observed in this electrocardiogram that we haven't discussed?"
  - [0.31] ❌ "Identify any additional electrical pattern variations in this ECG."

### ORIGINAL: "Is there early repolarization?"

  - [1.00] ✅ MATCH: "Is there early repolarization?"
  - [0.75] ✅ MATCH: "Is there any early repolarization pattern visible?"
  - [0.70] ✅ MATCH: "Is a pattern of early repolarization identifiable?"
  - [0.70] ✅ MATCH: "Is there a classic early repolarization pattern visible?"
  - [0.69] ✅ MATCH: "Does this tracing demonstrate early repolarization?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Describe any peripheral ECG alterations you note in this tracing."
  - [0.35] ❌ "Check for any unusual T wave configurations."

### ORIGINAL: "Are there any nonspecific ST-T changes?"

  - [1.00] ✅ MATCH: "Are there any nonspecific ST-T changes?"
  - [0.84] ✅ MATCH: "Are there any subtle nonspecific ST-T changes present?"
  - [0.79] ✅ MATCH: "Please identify any nonspecific ST-T changes."
  - [0.76] ✅ MATCH: "Evaluate for any other nonspecific ST-T changes."
  - [0.76] ✅ MATCH: "Assess whether any nonspecific ST-T changes are present."
  **WORST MATCHES (potential misassignments):**
  - [0.30] ❌ "Look closely for any marginal ST-T segment modifications in this ECG."
  - [0.31] ❌ "Can minor ST segment shifts be identified in this tracing?"

### ORIGINAL: "What other findings are present?"

  - [1.00] ✅ MATCH: "What other findings are present?"
  - [0.79] ✅ MATCH: "What additional ECG findings are present?"
  - [0.77] ✅ MATCH: "What other ECG findings are present in this record?"
  - [0.71] ✅ MATCH: "What other findings beyond the main diagnosis are present?"
  - [0.70] ✅ MATCH: "What further findings are apparent on this tracing?"

### ORIGINAL: "Are the T waves normal in morphology?"

  - [1.00] ✅ MATCH: "Are the T waves normal in morphology?"
  - [0.79] ✅ MATCH: "Do the T waves exhibit normal morphology?"
  - [0.77] ✅ MATCH: "Would you say the T waves appear normal in morphology?"
  - [0.73] ✅ MATCH: "Are the T waves showing normal morphological pattern?"
  - [0.72] ✅ MATCH: "Are the T waves showing normal morphology in this record?"
  **WORST MATCHES (potential misassignments):**
  - [0.27] ❌ "What incidental electrical patterns appear in this tracing?"
  - [0.28] ❌ "What minor cardiac electrical shifts can be observed in this tracing?"


## category_pericarditis (5 original prompts, 402 variations)

### ORIGINAL: "Is there diffuse ST elevation?"

  - [1.00] ✅ MATCH: "Is there diffuse ST elevation?"
  - [0.64] ✅ MATCH: "Please assess whether diffuse ST elevation is present."
  - [0.59] ✅ MATCH: "Assess the ST segments for diffuse elevation pattern"
  - [0.56] ✅ MATCH: "Check for the characteristic diffuse ST elevation of pericarditis."
  - [0.56] ✅ MATCH: "Is there a characteristic concave ST elevation pattern present?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Assess the ST elevation pattern and its distribution across the precordium."
  - [0.39] ❌ "What pattern of ST elevation do you observe in the precordial leads?"

### ORIGINAL: "Are there pericarditic changes?"

  - [1.00] ✅ MATCH: "Are there pericarditic changes?"
  - [0.72] ✅ MATCH: "Examine the tracing for pericarditic ST changes"
  - [0.71] ✅ MATCH: "What is your assessment of these pericarditic changes?"
  - [0.70] ✅ MATCH: "Evaluate the ST segments for pericarditic changes"
  - [0.70] ✅ MATCH: "Look for the presence of pericarditic ECG changes"

### ORIGINAL: "Are there signs of pericardial inflammation?"

  - [1.00] ✅ MATCH: "Are there signs of pericardial inflammation?"
  - [0.80] ✅ MATCH: "Please evaluate for signs of pericardial inflammation"
  - [0.79] ✅ MATCH: "Assess this tracing for signs of pericardial inflammation"
  - [0.75] ✅ MATCH: "Are there diffuse changes suggesting pericardial inflammation?"
  - [0.75] ✅ MATCH: "Are the ST changes seen here suggestive of pericardial inflammation?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "What ST segment abnormalities do you see in this tracing?"
  - [0.38] ❌ "How would you characterize the ST segment abnormalities in this tracing?"

### ORIGINAL: "Does this ECG suggest pericarditis?"

  - [1.00] ✅ MATCH: "Does this ECG suggest pericarditis?"
  - [0.78] ✅ MATCH: "What does this ECG suggest about pericardial status?"
  - [0.76] ✅ MATCH: "Does this ECG demonstrate pericarditic changes?"
  - [0.74] ✅ MATCH: "Does this pattern suggest pericardial inflammation?"
  - [0.71] ✅ MATCH: "Does the ST segment morphology suggest pericarditis?"
  **WORST MATCHES (potential misassignments):**
  - [0.31] ❌ "What ST segment abnormalities are seen in this tracing?"
  - [0.38] ❌ "Describe the ST segment abnormalities and their significance."

### ORIGINAL: "Is there evidence of pericarditis?"

  - [1.00] ✅ MATCH: "Is there evidence of pericarditis?"
  - [0.78] ✅ MATCH: "Is there evidence of pericarditis in these waveforms?"
  - [0.76] ✅ MATCH: "Assess the tracing for evidence of pericarditis."
  - [0.71] ✅ MATCH: "Evaluate this tracing for evidence of pericarditis"
  - [0.69] ✅ MATCH: "Evaluate the ST segments for evidence of pericarditis"


## category_rhythm (9 original prompts, 600 variations)

### ORIGINAL: "What is the rhythm in this ECG?"

  - [1.00] ✅ MATCH: "What is the rhythm in this ECG?"
  - [0.85] ✅ MATCH: "What is the rhythm type shown in this ECG?"
  - [0.84] ✅ MATCH: "What is the rhythm status of this ECG?"
  - [0.84] ✅ MATCH: "What is the fundamental rhythm in this ECG?"
  - [0.82] ✅ MATCH: "What is the underlying rhythm on this ECG?"

### ORIGINAL: "What is the heart rate and rhythm?"

  - [1.00] ✅ MATCH: "What is the heart rate and rhythm?"
  - [0.62] ✅ MATCH: "What does this ECG reveal about the rhythm?"
  - [0.61] ✅ MATCH: "What is the cardiac rhythm classification?"
  - [0.53] ✅ MATCH: "Does this ECG represent a sinus-based rhythm?"
  - [0.52] ✅ MATCH: "Does this tracing demonstrate a regular rhythm?"

### ORIGINAL: "Are there any ectopic beats?"

  - [1.00] ✅ MATCH: "Are there any ectopic beats?"
  - [0.67] ✅ MATCH: "Are there any ectopic or abnormal rhythms present?"
  - [0.56] ✅ MATCH: "Does this ECG display an ectopic rhythm?"

### ORIGINAL: "Is the rhythm regular or irregular?"

  - [1.00] ✅ MATCH: "Is the rhythm regular or irregular?"
  - [0.85] ✅ MATCH: "Is the rhythm in this ECG regular or irregular?"
  - [0.82] ✅ MATCH: "Assess the rhythm status - regular or irregular?"
  - [0.78] ✅ MATCH: "State whether this rhythm is regular or irregular."
  - [0.77] ✅ MATCH: "Does the rhythm appear normal or irregular?"

### ORIGINAL: "Are there any rhythm abnormalities?"

  - [1.00] ✅ MATCH: "Are there any rhythm abnormalities?"
  - [0.88] ✅ MATCH: "Are there any rhythm abnormalities to report?"
  - [0.78] ✅ MATCH: "Does this ECG show any rhythm abnormalities?"
  - [0.76] ✅ MATCH: "Does this ECG display any rhythm abnormalities?"
  - [0.74] ✅ MATCH: "Does this tracing exhibit any rhythm abnormalities?"

### ORIGINAL: "Is there any arrhythmia present?"

  - [1.00] ✅ MATCH: "Is there any arrhythmia present?"
  - [0.84] ✅ MATCH: "Is there any arrhythmia present in this ECG?"
  - [0.76] ✅ MATCH: "Is there any arrhythmia visible in this ECG?"
  - [0.71] ✅ MATCH: "Is there any evidence of an arrhythmia here?"
  - [0.68] ✅ MATCH: "Is there a rhythm disorder present in this ECG?"

### ORIGINAL: "Is this sinus rhythm or something else?"

  - [1.00] ✅ MATCH: "Is this sinus rhythm or something else?"
  - [0.85] ✅ MATCH: "Is this rhythm sinus or something else?"
  - [0.79] ✅ MATCH: "Is this rhythm sinus-based or something else?"
  - [0.76] ✅ MATCH: "Is this tracing showing regular sinus rhythm or something else?"
  - [0.69] ✅ MATCH: "Is this a sinus rhythm or an abnormal rhythm?"

### ORIGINAL: "What type of rhythm is shown?"

  - [1.00] ✅ MATCH: "What type of rhythm is shown?"
  - [0.79] ✅ MATCH: "What type of heartbeat rhythm is shown here?"
  - [0.76] ✅ MATCH: "What type of heart rhythm is shown in this ECG?"
  - [0.75] ✅ MATCH: "Assess what type of rhythm is present."
  - [0.72] ✅ MATCH: "What type of heart rhythm is shown in this tracing?"

### ORIGINAL: "Can you identify the cardiac rhythm?"

  - [1.00] ✅ MATCH: "Can you identify the cardiac rhythm?"
  - [0.86] ✅ MATCH: "Can you identify the cardiac rhythm on this ECG?"
  - [0.77] ✅ MATCH: "Could you identify the cardiac rhythm from this ECG?"
  - [0.69] ✅ MATCH: "Please identify and name this cardiac rhythm."
  - [0.69] ✅ MATCH: "Can you describe the cardiac rhythm shown in this ECG?"


## classification (15 original prompts, 700 variations)

### ORIGINAL: "Is this ECG normal or abnormal?"

  - [1.00] ✅ MATCH: "Is this ECG normal or abnormal?"
  - [0.91] ✅ MATCH: "Is this tracing normal or abnormal?"
  - [0.88] ✅ MATCH: "Is this a normal or abnormal ECG?"
  - [0.84] ✅ MATCH: "Is this a normal or an abnormal ECG?"
  - [0.78] ✅ MATCH: "Is this ECG showing normal or abnormal patterns?"

### ORIGINAL: "Is this ECG pathological?"

  - [1.00] ✅ MATCH: "Is this ECG pathological?"
  - [0.61] ✅ MATCH: "Can this ECG be considered pathological or benign?"
  - [0.60] ✅ MATCH: "Is this a normal ECG or does it show pathological findings?"
  - [0.59] ✅ MATCH: "Can this ECG be described as healthy or pathological?"
  - [0.55] ✅ MATCH: "Is this ECG representative of normal physiology or disease?"

### ORIGINAL: "How would you classify this ECG?"

  - [1.00] ✅ MATCH: "How would you classify this ECG?"
  - [0.81] ✅ MATCH: "How do you classify this ECG tracing?"
  - [0.78] ✅ MATCH: "How would you evaluate this ECG?"
  - [0.76] ✅ MATCH: "How would you label this ECG result?"
  - [0.75] ✅ MATCH: "How would you label this ECG finding?"

### ORIGINAL: "Should I be concerned about this ECG?"

  - [1.00] ✅ MATCH: "Should I be concerned about this ECG?"
  - [0.68] ✅ MATCH: "What is your conclusion about this ECG?"
  - [0.65] ✅ MATCH: "What is your expert conclusion about this ECG?"
  - [0.60] ✅ MATCH: "Do you see any concerning patterns in this ECG?"
  - [0.59] ✅ MATCH: "What is the clinical verdict on this ECG?"

### ORIGINAL: "What is the overall ECG assessment?"

  - [1.00] ✅ MATCH: "What is the overall ECG assessment?"
  - [0.76] ✅ MATCH: "What is the overall assessment of this tracing?"
  - [0.64] ✅ MATCH: "What is the overall impression of this ECG study?"
  - [0.63] ✅ MATCH: "What is the overall classification of this ECG?"
  - [0.63] ✅ MATCH: "What is the overall classification of this ECG result?"

### ORIGINAL: "Are there any abnormal findings in this ECG?"

  - [1.00] ✅ MATCH: "Are there any abnormal findings in this ECG?"
  - [0.84] ✅ MATCH: "Are there any concerning findings in this ECG?"
  - [0.74] ✅ MATCH: "Are there any abnormal features visible in this tracing?"
  - [0.74] ✅ MATCH: "Is there any abnormality detectable in this ECG?"
  - [0.73] ✅ MATCH: "Are there any concerning findings in this electrocardiogram?"

### ORIGINAL: "Is this ECG concerning?"

  - [1.00] ✅ MATCH: "Is this ECG concerning?"
  - [0.69] ✅ MATCH: "Is this ECG showing any concerning patterns?"
  - [0.65] ✅ MATCH: "Does this ECG reveal any concerning findings?"
  - [0.63] ✅ MATCH: "Is this ECG concerning from a clinical standpoint?"
  - [0.61] ✅ MATCH: "Classify this ECG tracing."

### ORIGINAL: "Does this ECG require follow-up?"

  - [1.00] ✅ MATCH: "Does this ECG require follow-up?"
  - [0.58] ✅ MATCH: "What issues does this ECG reveal?"
  - [0.56] ✅ MATCH: "What pathology does this ECG reveal?"
  - [0.53] ✅ MATCH: "What does this ECG demonstrate about the heart?"
  - [0.52] ✅ MATCH: "What does this ECG reveal about cardiac function?"

### ORIGINAL: "Would you classify this ECG as normal or abnormal?"

  - [1.00] ✅ MATCH: "Would you classify this ECG as normal or abnormal?"
  - [0.92] ✅ MATCH: "Would you flag this ECG as normal or abnormal?"
  - [0.87] ✅ MATCH: "Can you classify this as normal or abnormal?"
  - [0.87] ✅ MATCH: "Would you classify this as a normal or abnormal tracing?"
  - [0.87] ✅ MATCH: "Would you consider this ECG normal or abnormal?"

### ORIGINAL: "Is this a normal ECG?"

  - [1.00] ✅ MATCH: "Is this a normal ECG?"
  - [0.84] ✅ MATCH: "Is this a normal ECG tracing?"
  - [0.64] ✅ MATCH: "Is this a normal or pathological ECG pattern?"
  - [0.62] ✅ MATCH: "Is this a normal cardiac electrical pattern?"
  - [0.58] ✅ MATCH: "Does this look like a normal tracing to you?"

### ORIGINAL: "Is urgent action needed for this ECG?"

  - [1.00] ✅ MATCH: "Is urgent action needed for this ECG?"
  - [0.62] ✅ MATCH: "What is your final classification for this ECG?"
  - [0.60] ✅ MATCH: "What is the final classification for this ECG?"
  - [0.58] ✅ MATCH: "What is your evaluation of this ECG study?"
  - [0.58] ✅ MATCH: "What heart condition is indicated by this ECG?"

### ORIGINAL: "Does this ECG show any pathology?"

  - [1.00] ✅ MATCH: "Does this ECG show any pathology?"
  - [0.77] ✅ MATCH: "Does this ECG show any abnormalities?"
  - [0.77] ✅ MATCH: "Does this ECG suggest cardiac pathology?"
  - [0.75] ✅ MATCH: "Does this ECG show normal sinus rhythm or pathology?"
  - [0.74] ✅ MATCH: "Does this ECG show a normal rhythm?"

### ORIGINAL: "Is this ECG within normal limits?"

  - [1.00] ✅ MATCH: "Is this ECG within normal limits?"
  - [0.91] ✅ MATCH: "Is this tracing within normal limits?"
  - [0.86] ✅ MATCH: "Is this ECG considered within normal limits?"
  - [0.82] ✅ MATCH: "Is this electrocardiogram within normal limits?"
  - [0.80] ✅ MATCH: "Is this ECG within normal limits or pathological?"

### ORIGINAL: "Is there anything wrong with this ECG?"

  - [1.00] ✅ MATCH: "Is there anything wrong with this ECG?"
  - [0.77] ✅ MATCH: "Is there anything to worry about in this ECG?"
  - [0.73] ✅ MATCH: "Is there cause for concern with this ECG?"
  - [0.73] ✅ MATCH: "Is there anything abnormal in this ECG readout?"
  - [0.73] ✅ MATCH: "Is there anything abnormal in this ECG tracing?"

### ORIGINAL: "Does this ECG show borderline changes?"

  - [1.00] ✅ MATCH: "Does this ECG show borderline changes?"
  - [0.69] ✅ MATCH: "Does this ECG show signs of ischemia?"
  - [0.66] ✅ MATCH: "Does this ECG show evidence of cardiac disease?"
  - [0.63] ✅ MATCH: "Does this ECG show signs of cardiac disease?"
  - [0.63] ✅ MATCH: "Does this ECG show any abnormal ST changes or T wave inversion?"


## culprit_artery (4 original prompts, 400 variations)

### ORIGINAL: "What is the location of the coronary occlusion?"

  - [1.00] ✅ MATCH: "What is the location of the coronary occlusion?"
  - [0.93] ✅ MATCH: "What is the location of the coronary artery occlusion?"
  - [0.85] ✅ MATCH: "Pinpoint the location of the coronary occlusion"
  - [0.74] ✅ MATCH: "Locate the coronary occlusion"
  - [0.74] ✅ MATCH: "Which vessel is the source of the acute coronary occlusion?"
  **WORST MATCHES (potential misassignments):**
  - [0.30] ❌ "Please analyze this ECG and state the blocked vessel."
  - [0.35] ❌ "Name the coronary vessel responsible for these changes."

### ORIGINAL: "Which vessel is the culprit for this acute coronary syndrome?"

  - [1.00] ✅ MATCH: "Which vessel is the culprit for this acute coronary syndrome?"
  - [0.85] ✅ MATCH: "Which vessel is the culprit in this acute coronary situation?"
  - [0.84] ✅ MATCH: "Which coronary branch is the culprit for this acute coronary syndrome?"
  - [0.77] ✅ MATCH: "What is the culprit vessel in this acute coronary syndrome ECG?"
  - [0.75] ✅ MATCH: "Identify the culprit vessel in this acute coronary syndrome"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Determine the blocked artery from the lead-specific ST deviations."
  - [0.34] ❌ "Name the occluded vessel from the ST-segment distribution."

### ORIGINAL: "What is the culprit artery?"

  - [1.00] ✅ MATCH: "What is the culprit artery?"
  - [0.76] ✅ MATCH: "What is the culprit coronary vessel?"
  - [0.67] ✅ MATCH: "What is the culprit vessel in this ECG?"
  - [0.66] ✅ MATCH: "What is the culprit vessel in this case?"
  - [0.66] ✅ MATCH: "Which coronary is the culprit vessel?"

### ORIGINAL: "Which coronary artery is occluded?"

  - [1.00] ✅ MATCH: "Which coronary artery is occluded?"
  - [0.96] ✅ MATCH: "Which coronary artery has occluded?"
  - [0.89] ✅ MATCH: "Which coronary is occluded?"
  - [0.87] ✅ MATCH: "Identify which coronary artery is occluded"
  - [0.85] ✅ MATCH: "Which coronary branch is occluded?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Please specify which coronary branch is blocked, given the distribution of ST abnormalities in this tracing."
  - [0.36] ❌ "What artery has compromised perfusion causing these changes?"


## ecg_interval — SKIPPED (prompt-specific answers)


## interpretation (15 original prompts, 700 variations)

### ORIGINAL: "What are the key findings in this electrocardiogram?"

  - [1.00] ✅ MATCH: "What are the key findings in this electrocardiogram?"
  - [0.80] ✅ MATCH: "Are there any concerning findings in this electrocardiogram?"
  - [0.79] ✅ MATCH: "Please describe the findings on this electrocardiogram."
  - [0.77] ✅ MATCH: "What findings are notable in this electrocardiogram?"
  - [0.77] ✅ MATCH: "State your findings for this electrocardiogram."

### ORIGINAL: "What does this electrocardiogram reveal?"

  - [1.00] ✅ MATCH: "What does this electrocardiogram reveal?"
  - [0.85] ✅ MATCH: "What abnormalities does this electrocardiogram reveal?"
  - [0.82] ✅ MATCH: "What cardiac findings does this electrocardiogram reveal?"
  - [0.79] ✅ MATCH: "What does this electrocardiogram reveal about cardiac health?"
  - [0.76] ✅ MATCH: "What does this 12-lead ECG reveal?"

### ORIGINAL: "Can you read this ECG for me?"

  - [1.00] ✅ MATCH: "Can you read this ECG for me?"
  - [0.68] ✅ MATCH: "Can you read this ECG and tell me what you find?"
  - [0.63] ✅ MATCH: "Read and diagnose this ECG for me."
  - [0.61] ✅ MATCH: "Can you read this tracing and identify any issues?"
  - [0.60] ✅ MATCH: "Can you review this ECG and provide a diagnosis?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Examine this record for evidence of chamber enlargement or hypertrophy."
  - [0.38] ❌ "Look at this ECG and tell me if there are any concerns."

### ORIGINAL: "Please provide a complete ECG interpretation."

  - [1.00] ✅ MATCH: "Please provide a complete ECG interpretation."
  - [0.89] ✅ MATCH: "Please perform a complete ECG interpretation."
  - [0.82] ✅ MATCH: "Provide a comprehensive ECG interpretation."
  - [0.80] ✅ MATCH: "Please give a detailed ECG interpretation."
  - [0.75] ✅ MATCH: "Please perform a complete cardiac rhythm interpretation."
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Assess the relationship between P waves and QRS complexes."
  - [0.37] ❌ "Examine this tracing and identify any signs of ischemia or infarction."

### ORIGINAL: "What does this ECG show?"

  - [1.00] ✅ MATCH: "What does this ECG show?"
  - [0.84] ✅ MATCH: "What findings does this ECG show?"
  - [0.72] ✅ MATCH: "What rhythm abnormality does this ECG show?"
  - [0.68] ✅ MATCH: "What does the rhythm strip show?"
  - [0.68] ✅ MATCH: "What rhythm does this ECG demonstrate?"

### ORIGINAL: "Can you provide an ECG interpretation?"

  - [1.00] ✅ MATCH: "Can you provide an ECG interpretation?"
  - [0.80] ✅ MATCH: "Can you perform an ECG interpretation for me?"
  - [0.76] ✅ MATCH: "Can you provide a detailed interpretation of this ECG?"
  - [0.75] ✅ MATCH: "Can you provide a rhythm interpretation for this ECG?"
  - [0.73] ✅ MATCH: "Can you provide a clinical interpretation of this tracing?"

### ORIGINAL: "What is your analysis of this ECG?"

  - [1.00] ✅ MATCH: "What is your analysis of this ECG?"
  - [0.96] ✅ MATCH: "What's your analysis of this ECG?"
  - [0.77] ✅ MATCH: "What is your diagnosis for this ECG tracing?"
  - [0.72] ✅ MATCH: "Give me your expert analysis of this ECG."
  - [0.72] ✅ MATCH: "Provide your expert analysis of this ECG."

### ORIGINAL: "What abnormalities are present in this ECG?"

  - [1.00] ✅ MATCH: "What abnormalities are present in this ECG?"
  - [0.91] ✅ MATCH: "What cardiac abnormalities are present in this ECG?"
  - [0.91] ✅ MATCH: "What abnormalities are evident on this ECG?"
  - [0.88] ✅ MATCH: "What abnormalities are present in this heart recording?"
  - [0.86] ✅ MATCH: "What abnormalities are present in this 12-lead recording?"

### ORIGINAL: "What are the findings in this ECG?"

  - [1.00] ✅ MATCH: "What are the findings in this ECG?"
  - [0.94] ✅ MATCH: "What are the key findings in this ECG?"
  - [0.84] ✅ MATCH: "What are the key findings in this 12-lead?"
  - [0.84] ✅ MATCH: "What are the ECG findings in this patient?"
  - [0.74] ✅ MATCH: "What is the underlying rhythm in this ECG?"

### ORIGINAL: "Describe all ECG abnormalities present."

  - [1.00] ✅ MATCH: "Describe all ECG abnormalities present."
  - [0.73] ✅ MATCH: "Describe any ST segment abnormalities."
  - [0.69] ✅ MATCH: "Describe the abnormalities seen in this tracing."
  - [0.63] ✅ MATCH: "Review this ECG and describe all abnormalities."
  - [0.61] ✅ MATCH: "Identify and describe all abnormalities in this ECG tracing."

### ORIGINAL: "What do you see in this ECG recording?"

  - [1.00] ✅ MATCH: "What do you see in this ECG recording?"
  - [0.76] ✅ MATCH: "Tell me what you see in this ECG tracing."
  - [0.75] ✅ MATCH: "Describe what you see in this ECG tracing."
  - [0.67] ✅ MATCH: "What does this heart recording indicate?"
  - [0.65] ✅ MATCH: "What abnormal patterns do you observe in this tracing?"

### ORIGINAL: "What is your ECG diagnosis?"

  - [1.00] ✅ MATCH: "What is your ECG diagnosis?"
  - [0.70] ✅ MATCH: "Can you give an ECG diagnosis?"
  - [0.54] ✅ MATCH: "Examine this ECG and provide your diagnosis."
  - [0.51] ✅ MATCH: "Examine this ECG and tell me the diagnosis."
  - [0.49] ✅ MATCH: "State what this ECG pattern represents diagnostically."

### ORIGINAL: "Please describe the ECG findings."

  - [1.00] ✅ MATCH: "Please describe the ECG findings."
  - [0.69] ✅ MATCH: "Summarize the key ECG findings."
  - [0.68] ✅ MATCH: "Please read this ECG tracing."
  - [0.67] ✅ MATCH: "Please describe the ECG abnormalities in clinical terms."
  - [0.67] ✅ MATCH: "Please examine this ECG and state your findings."
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "Assess the voltage criteria for ventricular hypertrophy."
  - [0.34] ❌ "Examine the T waves and tell me what you observe."

### ORIGINAL: "Can you analyze this ECG tracing?"

  - [1.00] ✅ MATCH: "Can you analyze this ECG tracing?"
  - [0.87] ✅ MATCH: "Can you evaluate this ECG tracing?"
  - [0.77] ✅ MATCH: "Can you diagnose this ECG strip?"
  - [0.76] ✅ MATCH: "Could you analyze this electrocardiogram tracing?"
  - [0.72] ✅ MATCH: "Can you analyze this ECG and identify issues?"

### ORIGINAL: "Can you interpret this ECG?"

  - [1.00] ✅ MATCH: "Can you interpret this ECG?"
  - [0.87] ✅ MATCH: "Can you interpret this 12-lead ECG?"
  - [0.71] ✅ MATCH: "Can you give a reading of this ECG?"
  - [0.71] ✅ MATCH: "Can you determine what this ECG shows?"
  - [0.70] ✅ MATCH: "Could you interpret this ECG tracing for me?"


## json_interpretation (5 original prompts, 399 variations)

### ORIGINAL: "Output JSON ONLY with keys: RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification. Values must be lists of present findings (omit missing categories)."

  - [0.64] ✅ MATCH: "Output JSON ONLY with keys: {keys}. Values must be lists of present findings (omit missing categories)."

### ORIGINAL: "Return ECG findings as JSON with keys RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification."

  - [0.43] ✅ MATCH: "Return ECG findings as JSON with keys {keys}."
  - [0.43] ✅ MATCH: "Return a JSON document with keys {keys} representing the tracing."
  - [0.41] ✅ MATCH: "Return a JSON object with {keys} representing this ECG."
  - [0.41] ✅ MATCH: "Return JSON with keys {keys} representing abnormalities."
  - [0.40] ❌ MISMATCH: "Return JSON with keys {keys} representing ECG findings."
  **WORST MATCHES (potential misassignments):**
  - [0.19] ❌ "Can you construct JSON with the following structure: {keys}?"
  - [0.22] ❌ "What JSON keys {keys} would you use for this ECG?"

### ORIGINAL: "Generate JSON representation of ECG abnormalities. Keys: RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification."

  - [0.52] ✅ MATCH: "Generate JSON representation of ECG abnormalities. Keys: {keys}."
  - [0.41] ✅ MATCH: "Generate a JSON representation of the ECG findings. Include keys {keys}."
  - [0.41] ✅ MATCH: "Create JSON representation with keys {keys} for this tracing."
  - [0.41] ✅ MATCH: "Build JSON representation with {keys} for this heart tracing."
  - [0.40] ✅ MATCH: "Generate JSON representation of this ECG. Use keys {keys}."
  **WORST MATCHES (potential misassignments):**
  - [0.22] ❌ "Can you encode the heart rhythm as JSON using {keys}?"
  - [0.25] ❌ "Can you generate JSON with these keys: {keys}?"

### ORIGINAL: "Provide structured JSON output for this ECG analysis using keys RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification."

  - [0.55] ✅ MATCH: "Provide structured JSON output for this ECG analysis using keys {keys}."
  - [0.46] ✅ MATCH: "Provide structured JSON output for this ECG with keys {keys}."
  - [0.42] ✅ MATCH: "Provide structured JSON using keys {keys} for this tracing."
  - [0.39] ❌ MISMATCH: "Provide JSON output for this ECG tracing. Include keys {keys}."
  - [0.39] ❌ MISMATCH: "Return a JSON object for this ECG tracing. Keys must include {keys}."
  **WORST MATCHES (potential misassignments):**
  - [0.22] ❌ "Format this heart rhythm as JSON. Required keys: {keys}."
  - [0.22] ❌ "Provide the cardiac assessment in JSON structure with {keys}."

### ORIGINAL: "Output ECG interpretation in JSON format only (keys RHYTHM, CONDUCTION, CHAMBER_ENLARGEMENT, INFARCT_ISCHEMIA, PERICARDITIS, heart_rate_bpm, ecg_classification)."

  - [0.51] ✅ MATCH: "Output ECG interpretation in JSON format only (keys {keys})."
  - [0.42] ✅ MATCH: "Return ECG interpretation in JSON format. Keys: {keys}."
  - [0.39] ❌ MISMATCH: "Provide ECG interpretation in JSON form. Keys: {keys}."
  - [0.37] ❌ MISMATCH: "Output ECG findings in JSON. Keys should include {keys}."
  - [0.37] ❌ MISMATCH: "Submit ECG interpretation as JSON data with keys {keys}."
  **WORST MATCHES (potential misassignments):**
  - [0.20] ❌ "What JSON would you generate with keys {keys}?"
  - [0.25] ❌ "I'd like the ECG results in JSON. Include {keys}."


## localization_q_wave (5 original prompts, 400 variations)

### ORIGINAL: "Are there Q waves suggesting old infarction?"

  - [1.00] ✅ MATCH: "Are there Q waves suggesting old infarction?"
  - [0.90] ✅ MATCH: "Are there Q waves suggesting previous infarction?"
  - [0.87] ✅ MATCH: "Are there Q waves suggesting old anteroseptal infarction?"
  - [0.78] ✅ MATCH: "Are there any Q waves that suggest old myocardial infarction?"
  - [0.77] ✅ MATCH: "Is there evidence of Q wave changes suggesting old infarction?"

### ORIGINAL: "Where are Q waves located?"

  - [1.00] ✅ MATCH: "Where are Q waves located?"
  - [0.81] ✅ MATCH: "Where are the Q waves situated?"
  - [0.75] ✅ MATCH: "Where are the Q waves located anatomically?"
  - [0.73] ✅ MATCH: "Where on the 12-lead ECG are Q waves located?"
  - [0.72] ✅ MATCH: "Where are Q waves located in this 12-lead ECG?"

### ORIGINAL: "Are there any Q waves present?"

  - [1.00] ✅ MATCH: "Are there any Q waves present?"
  - [0.72] ✅ MATCH: "Please report any Q wave presence."
  - [0.71] ✅ MATCH: "Assess whether Q waves are present."
  - [0.70] ✅ MATCH: "Determine if Q waves are present."
  - [0.70] ✅ MATCH: "Examine whether Q waves are present."
  **WORST MATCHES (potential misassignments):**
  - [0.31] ❌ "Find Q wave abnormalities in the anterior precordial leads."
  - [0.33] ❌ "Find Q wave abnormalities across the precordial leads."

### ORIGINAL: "Can you identify pathological Q waves?"

  - [1.00] ✅ MATCH: "Can you identify pathological Q waves?"
  - [0.86] ✅ MATCH: "Can you identify pathological Q waves in this ECG?"
  - [0.73] ✅ MATCH: "Can you detect pathological Q waves in this tracing?"
  - [0.72] ✅ MATCH: "What leads contain pathological Q waves?"
  - [0.70] ✅ MATCH: "Can pathological Q waves be seen?"

### ORIGINAL: "Is there evidence of Q waves?"

  - [1.00] ✅ MATCH: "Is there evidence of Q waves?"
  - [0.83] ✅ MATCH: "Is there evidence of Q waves in any lead?"
  - [0.75] ✅ MATCH: "Examine the ECG for evidence of Q waves."
  - [0.71] ✅ MATCH: "Can you find evidence of Q waves?"
  - [0.70] ✅ MATCH: "Is there evidence of Q waves in the lateral territory?"


## localization_qrs_axis (7 original prompts, 400 variations)

### ORIGINAL: "Is there axis deviation?"

  - [1.00] ✅ MATCH: "Is there axis deviation?"
  - [0.80] ✅ MATCH: "Is there any axis deviation present?"
  - [0.77] ✅ MATCH: "What is the axis deviation angle?"
  - [0.76] ✅ MATCH: "How extreme is the axis deviation?"
  - [0.75] ✅ MATCH: "Review the axis for deviation"

### ORIGINAL: "What is the frontal plane QRS axis?"

  - [1.00] ✅ MATCH: "What is the frontal plane QRS axis?"
  - [0.93] ✅ MATCH: "What is the frontal plane mean QRS axis?"
  - [0.87] ✅ MATCH: "What is the mean frontal plane axis?"
  - [0.86] ✅ MATCH: "What is the frontal plane electrical axis?"
  - [0.85] ✅ MATCH: "Assess the frontal plane QRS axis"

### ORIGINAL: "What type of axis deviation is present?"

  - [1.00] ✅ MATCH: "What type of axis deviation is present?"
  - [0.88] ✅ MATCH: "What type of axis deviation is seen here?"
  - [0.86] ✅ MATCH: "What type of axis shift is present?"
  - [0.84] ✅ MATCH: "What type of cardiac axis deviation is seen?"
  - [0.78] ✅ MATCH: "Identify the type of axis deviation present"

### ORIGINAL: "Is there left or right axis deviation?"

  - [1.00] ✅ MATCH: "Is there left or right axis deviation?"
  - [0.86] ✅ MATCH: "Is there any left or right axis deviation present?"
  - [0.85] ✅ MATCH: "Does the QRS show left or right axis deviation?"
  - [0.80] ✅ MATCH: "Does the QRS complex show left or right axis deviation?"
  - [0.78] ✅ MATCH: "Is this ECG showing right axis deviation?"

### ORIGINAL: "Is the QRS axis normal or deviated?"

  - [1.00] ✅ MATCH: "Is the QRS axis normal or deviated?"
  - [0.89] ✅ MATCH: "Is the cardiac axis normal or deviated?"
  - [0.86] ✅ MATCH: "Does the QRS axis appear normal or deviated?"
  - [0.81] ✅ MATCH: "Is the axis normal or left/right deviated?"
  - [0.78] ✅ MATCH: "Is the axis in the normal range or deviated?"

### ORIGINAL: "What is the QRS axis?"

  - [1.00] ✅ MATCH: "What is the QRS axis?"
  - [0.84] ✅ MATCH: "What is the QRS frontal axis?"
  - [0.82] ✅ MATCH: "What is the measured QRS axis?"
  - [0.78] ✅ MATCH: "What is the QRS axis measurement?"
  - [0.75] ✅ MATCH: "What does the QRS axis indicate?"

### ORIGINAL: "What is the electrical axis of the heart?"

  - [1.00] ✅ MATCH: "What is the electrical axis of the heart?"
  - [0.87] ✅ MATCH: "What is the electrical axis of the ventricles?"
  - [0.86] ✅ MATCH: "Is the electrical axis of the heart normal?"
  - [0.85] ✅ MATCH: "What is the electrical axis on this 12-lead?"
  - [0.84] ✅ MATCH: "What is the electrical axis of the QRS complex?"


## localization_st_depression (5 original prompts, 400 variations)

### ORIGINAL: "Are there ST depressions in any leads?"

  - [1.00] ✅ MATCH: "Are there ST depressions in any leads?"
  - [0.90] ✅ MATCH: "Is there ST depression in any lead?"
  - [0.81] ✅ MATCH: "Are there any ST depressions in the inferior leads?"
  - [0.80] ✅ MATCH: "Is there localized ST depression in any lead?"
  - [0.78] ✅ MATCH: "Is there ST depression in the lateral leads?"

### ORIGINAL: "Where is ST depression located?"

  - [1.00] ✅ MATCH: "Where is ST depression located?"
  - [0.89] ✅ MATCH: "Where exactly is ST depression located?"
  - [0.79] ✅ MATCH: "Where does ST depression appear?"
  - [0.76] ✅ MATCH: "Where is the ST depression located on this tracing?"
  - [0.72] ✅ MATCH: "Please report where ST depression is located"

### ORIGINAL: "Is there evidence of ST depression?"

  - [1.00] ✅ MATCH: "Is there evidence of ST depression?"
  - [0.85] ✅ MATCH: "Is there evidence of ST depression in this ECG?"
  - [0.76] ✅ MATCH: "Investigate the presence of ST depression"
  - [0.76] ✅ MATCH: "Is there evidence of ST depression in the anterior leads?"
  - [0.75] ✅ MATCH: "What is the pattern of ST depression?"

### ORIGINAL: "Is there ST depression?"

  - [1.00] ✅ MATCH: "Is there ST depression?"
  - [0.79] ✅ MATCH: "Is there any ST depression visible?"
  - [0.74] ✅ MATCH: "Is there any ST depression in this ECG?"
  - [0.74] ✅ MATCH: "Which leads have ST depression?"
  - [0.71] ✅ MATCH: "Find the leads with ST depression"

### ORIGINAL: "Can you identify ST segment depression?"

  - [1.00] ✅ MATCH: "Can you identify ST segment depression?"
  - [0.81] ✅ MATCH: "Identify any ST segment depressions"
  - [0.81] ✅ MATCH: "Can you find the ST segment depressions?"
  - [0.80] ✅ MATCH: "Can you find ST segment depressions here?"
  - [0.79] ✅ MATCH: "Can you identify any ST segment depressions in this tracing?"


## localization_t_wave (5 original prompts, 400 variations)

### ORIGINAL: "Can you identify T wave changes?"

  - [1.00] ✅ MATCH: "Can you identify T wave changes?"
  - [0.83] ✅ MATCH: "Would you identify any T wave changes?"
  - [0.79] ✅ MATCH: "Can you identify T wave inversion here?"
  - [0.76] ✅ MATCH: "Can you identify any T wave changes in this tracing?"
  - [0.71] ✅ MATCH: "Can you identify the inverted T waves?"

### ORIGINAL: "Is there T wave inversion present?"

  - [1.00] ✅ MATCH: "Is there T wave inversion present?"
  - [0.83] ✅ MATCH: "Show me where T wave inversion is present"
  - [0.80] ✅ MATCH: "What leads have T wave inversion present?"
  - [0.78] ✅ MATCH: "Is there T wave inversion in this trace?"
  - [0.78] ✅ MATCH: "Report any T wave inversion present"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Find the specific lead locations of inverted T waves"
  - [0.35] ❌ "Find the specific leads with inverted T waves"

### ORIGINAL: "Where are T wave inversions?"

  - [1.00] ✅ MATCH: "Where are T wave inversions?"
  - [0.88] ✅ MATCH: "Where are T wave inversions located?"
  - [0.85] ✅ MATCH: "Where are the T wave inversions found?"
  - [0.85] ✅ MATCH: "Where is T wave inversion seen?"
  - [0.84] ✅ MATCH: "Where are T wave inversions identified?"

### ORIGINAL: "Are the T waves abnormal in any leads?"

  - [1.00] ✅ MATCH: "Are the T waves abnormal in any leads?"
  - [0.84] ✅ MATCH: "Are the T waves inverted in any leads?"
  - [0.69] ✅ MATCH: "Is the T wave inverted in any of the leads?"
  - [0.69] ✅ MATCH: "Are inverted T waves visible in any of the leads?"
  - [0.69] ✅ MATCH: "Review the T waves in every lead"

### ORIGINAL: "Are there T wave abnormalities?"

  - [1.00] ✅ MATCH: "Are there T wave abnormalities?"
  - [0.79] ✅ MATCH: "Can you describe the T wave abnormalities?"
  - [0.78] ✅ MATCH: "Are there any T wave abnormalities worth noting?"
  - [0.77] ✅ MATCH: "Locate any T wave abnormalities"
  - [0.76] ✅ MATCH: "Is there any T wave abnormality present?"


## lvef (5 original prompts, 400 variations)

### ORIGINAL: "Can you tell me the patient's EF?"

  - [1.00] ✅ MATCH: "Can you tell me the patient's EF?"
  - [0.81] ✅ MATCH: "Can you tell me the patient's ejection fraction?"
  - [0.61] ✅ MATCH: "Can you determine the LVEF value?"
  - [0.60] ✅ MATCH: "Can you state the EF value?"
  - [0.60] ✅ MATCH: "Can you give me the EF data point?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Report the systolic performance percentage."
  - [0.34] ❌ "Provide the systolic performance percentage."

### ORIGINAL: "What is the LVEF based on echocardiography?"

  - [1.00] ✅ MATCH: "What is the LVEF based on echocardiography?"
  - [0.69] ✅ MATCH: "What is the calculated EF from the echocardiogram?"
  - [0.66] ✅ MATCH: "What EF did the echocardiogram show?"
  - [0.58] ✅ MATCH: "What EF value appears in the echocardiography report?"
  - [0.58] ✅ MATCH: "What is the measured EF value?"

### ORIGINAL: "What is the left ventricular function?"

  - [1.00] ✅ MATCH: "What is the left ventricular function?"
  - [0.94] ✅ MATCH: "What is the left ventricular pump function?"
  - [0.89] ✅ MATCH: "What is the left ventricular systolic function?"
  - [0.89] ✅ MATCH: "What is the left ventricular EF?"
  - [0.88] ✅ MATCH: "What is the documented ventricular function?"

### ORIGINAL: "What is the patient's left ventricular ejection fraction?"

  - [1.00] ✅ MATCH: "What is the patient's left ventricular ejection fraction?"
  - [0.89] ✅ MATCH: "What is the calculated left ventricular ejection fraction?"
  - [0.87] ✅ MATCH: "What is the left ventricular ejection ratio?"
  - [0.86] ✅ MATCH: "What is the patient's left ventricular systolic function?"
  - [0.85] ✅ MATCH: "What is the left ventricular ejection fraction value?"

### ORIGINAL: "What is the ejection fraction?"

  - [1.00] ✅ MATCH: "What is the ejection fraction?"
  - [0.90] ✅ MATCH: "What is the ejection fraction metric?"
  - [0.90] ✅ MATCH: "What is the ejection fraction result?"
  - [0.87] ✅ MATCH: "What is the measured ejection fraction?"
  - [0.87] ✅ MATCH: "What is the LV ejection fraction value?"
  **WORST MATCHES (potential misassignments):**
  - [0.33] ❌ "Please report the cardiac performance percentage."
  - [0.34] ❌ "Report the cardiac systolic output value."


## random_finding_question — SKIPPED (prompt-specific answers)


## structural_heart_disease (3 original prompts, 400 variations)

### ORIGINAL: "Does this patient have structural heart disease?"

  - [1.00] ✅ MATCH: "Does this patient have structural heart disease?"
  - [0.91] ✅ MATCH: "Does this patient exhibit structural heart disease?"
  - [0.84] ✅ MATCH: "Does this echo reveal structural heart disease?"
  - [0.83] ✅ MATCH: "Could this patient have underlying structural heart disease?"
  - [0.82] ✅ MATCH: "Does the patient have structural heart pathology?"

### ORIGINAL: "Based on echocardiography data, does the patient have structural heart disease?"

  - [1.00] ✅ MATCH: "Based on echocardiography data, does the patient have structural heart disease?"
  - [0.84] ✅ MATCH: "Based on the echo, does this patient have structural heart disease?"
  - [0.68] ✅ MATCH: "Are there echocardiographic signs of structural heart disease?"
  - [0.65] ✅ MATCH: "Check the echocardiogram for structural heart disease."
  - [0.61] ✅ MATCH: "Assess the echocardiographic findings for structural disease."
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "What cardiac anatomical abnormalities are evident?"
  - [0.35] ❌ "What cardiac morphological abnormalities exist?"

### ORIGINAL: "Is there evidence of structural heart disease in this patient?"

  - [1.00] ✅ MATCH: "Is there evidence of structural heart disease in this patient?"
  - [0.81] ✅ MATCH: "Are there signs of structural heart disease in this ECG?"
  - [0.78] ✅ MATCH: "Is there evidence of structural cardiac disease?"
  - [0.77] ✅ MATCH: "Assess for structural heart disease in this patient."
  - [0.77] ✅ MATCH: "Is structural heart disease present in this patient?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "What morphological abnormalities are seen in the cardiac chambers?"
  - [0.39] ❌ "What structural cardiac abnormalities does the echocardiogram show?"


## urgency_assessment (3 original prompts, 400 variations)

### ORIGINAL: "Is this an emergency ECG finding?"

  - [1.00] ✅ MATCH: "Is this an emergency ECG finding?"
  - [0.78] ✅ MATCH: "Is this an urgent cardiac finding?"
  - [0.71] ✅ MATCH: "Is this a "must-act-now" ECG finding?"
  - [0.70] ✅ MATCH: "Would you consider this an emergency cardiac finding?"
  - [0.70] ✅ MATCH: "Is this a routine or urgent finding?"
  **WORST MATCHES (potential misassignments):**
  - [0.34] ❌ "How urgent is this finding - what's the time frame for action?"
  - [0.34] ❌ "Would you classify this as a concerning vs non-concerning ECG?"

### ORIGINAL: "What is the clinical urgency of this ECG?"

  - [1.00] ✅ MATCH: "What is the clinical urgency of this ECG?"
  - [0.88] ✅ MATCH: "What's the clinical urgency ranking of this ECG?"
  - [0.87] ✅ MATCH: "What's the clinical urgency of this finding?"
  - [0.79] ✅ MATCH: "What's the clinical urgency level for this ECG finding?"
  - [0.79] ✅ MATCH: "What is the clinical urgency level of the findings on this ECG?"
  **WORST MATCHES (potential misassignments):**
  - [0.32] ❌ "Are there any time-sensitive abnormalities that need rapid management?"
  - [0.34] ❌ "Are there any life-threatening patterns that need quick response?"

### ORIGINAL: "Does this ECG require immediate intervention?"

  - [1.00] ✅ MATCH: "Does this ECG require immediate intervention?"
  - [0.79] ✅ MATCH: "Does this require immediate clinical action?"
  - [0.77] ✅ MATCH: "Does this require immediate clinical escalation?"
  - [0.77] ✅ MATCH: "Does this require STAT interpretation?"
  - [0.76] ✅ MATCH: "Does this ECG warrant emergent intervention?"
  **WORST MATCHES (potential misassignments):**
  - [0.25] ❌ "Should this tracing trigger the rapid response team or standard protocols?"
  - [0.31] ❌ "Is this tracing concerning enough to warrant stat notification of the provider?"
