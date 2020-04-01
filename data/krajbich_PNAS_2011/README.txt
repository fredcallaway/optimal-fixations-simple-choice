————————————————————————————————————————Data——————————————————————————————————————————————
Files: ET3_for_sudeep (included as .csv and .RData)

Contains processed data for each subject and each trial of the experiment. Each line represents a new fixation.
N = 30
Choice trials with missing fixations for more than 500 ms at the beginning or end of the trial were excluded from analysis. Blank fixations are already accounted for as specified in the paper.

trial 		= trial number from 1 to 100
rating1 	= rating for the left item from 1 to 10
rating2 	= rating for the middle item from 1 to 10
rating3 	= rating for the right item from 1 to 10
roirating 	= rating for the currently fixated item
rt 		= reaction time for the trial (in milliseconds)
chosenrating 	= rating for the chosen item from 1 to 10
subject 	= participant ID #
eventduration 	= duration of the current fixation (in milliseconds)
fix_num 	= the fixation number within the current trial (e.g. 1 = first fixation, 2 								= second fixation, etc.)
choice1 	= dummy variable for if the left item was chosen (0 if no, 1 if yes)
choice2 	= dummy variable for if the middle item was chosen (0 if no, 1 if yes)
choice3		= dummy variable for if the right item was chosen (0 if no, 1 if yes)
leftroi 	= dummy variable for whether the current fixation is to the left item
middleroi 	= dummy variable for whether the current fixation is to the middle item
rightroi 	= dummy variable for whether the current fixation is to the right item
num_fixations 	= total number of fixations in the current trial
rev_fix_num 	= fixation number prior to the last fixation in the trial (e.g. 1 = last 					fixation, 2 = second-to-last fixation, etc.)
——————————————————————————————————————————————————————————————————————————————————————————