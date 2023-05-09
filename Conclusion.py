'''
Conclusion:
`Demand History` is a very important variable  because of its high correlation with `Region Demand` therefore showing high Dependancy for the latter.
'''

score = [DT_SC,RF_SC,XGB_SC,LR_SC]
Models = pd.DataFrame({
    'n_neighbors': ["Decision Tree","Random Forest","XGBoost", "Logistic Regression"],
    'Score': score})
Models.sort_values(by='Score', ascending=False)


''' 
Pretty much the entirety of this code was written with the help of online resources and other github libraries 
as this was one of my very first projects and i had a very little time to complete it :P

This is a very typical structure for supervised learning with several classes and all that I did was change a 
few variables to align it with the theme of our ideathon and it actually worked :D
'''