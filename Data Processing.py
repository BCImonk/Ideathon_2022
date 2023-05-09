##File Path:
# List all files under the input directory:
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Path for the training set:
tr_path = # TBD Later
# Path for the testing set:
te_path = # TBD Later



##Preprocessing and Data Analysis :
# Training Set:
# Read a csv file as a DataFrame:
tr_df = pd.read_csv(tr_path)
# Explore the first 5 rows:
tr_df.head()
# Testing Set:
# Read a csv file as a DataFrame:
te_df = pd.read_csv(te_path)
# Explore the first 5 rows:
te_df.head()
# Size of each data set:
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}")
# Column Information:
tr_df.info(verbose=True, null_counts=True)



##Data visalization:
'''We need to split our data to categorical and numerical data,
using the `.select_dtypes('dtype').columns.to_list()` combination.'''



##Region Demand Distribution:
# List of all the numeric columns:
num = tr_df.select_dtypes('number').columns.to_list()
# List of all the categoric columns:
cat = tr_df.select_dtypes('object').columns.to_list()
# Numeric df:
region_num =  tr_df[num]
# Categoric df:
region_cat = tr_df[cat]
print(tr_df[cat[-1]].value_counts())
total = float(len(tr_df[cat[-1]]))
plt.figure(figsize=(8,10))
sns.set(style="whitegrid")
ax = sns.countplot(tr_df[cat[-1]])
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center") 
plt.show()



##Let's plot our data:
#Numeric:
for i in region_num:
    plt.hist(region_num[i])
    plt.title(i)
    plt.show()
# Categorical (split by Region Demand):
for i in cat[:-1]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=i ,hue='Region_Status', data=tr_df ,palette='plasma')
    plt.xlabel(i, fontsize=14)



## Encoding data to numeric:
# Adding the new numeric values from the to_numeric variable to both datasets:
tr_df = tr_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable)
te_df = te_df.applymap(lambda lable: to_numeric.get(lable) if lable in to_numeric else lable
# Checking our manipulated dataset for validation
print(f"training set (row, col): {tr_df.shape}\n\ntesting set (row, col): {te_df.shape}\n")
print(tr_df.info(), "\n\n", te_df.info())
# Plotting the Correlation Matrix:
sns.heatmap(tr_df.corr() ,cmap='cubehelix_r')
# Correlation Table:
corr = tr_df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)