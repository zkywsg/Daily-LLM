[English](README_EN.md) | [中文](README.md)

# Machine Learning Foundations

> This chapter solves a concrete problem: when a consumer finance platform wants to predict whether a user will default within the next 90 days, how do you turn that business request into a trustworthy machine learning task?

## Learning Goals

By the end of this chapter, you should be able to answer:

1. How do you turn a real business problem into samples, features, labels, and evaluation metrics?
2. Why is logistic regression still a strong baseline in credit risk?
3. Why can high offline accuracy still lead to poor business outcomes?

## 1. From Business Problem to ML Task

Assume you work at a consumer finance platform. Each day, the system needs to decide which users can receive a higher credit line, which require stricter review, and which should trigger an early warning workflow.

If high-risk users are classified as low-risk, the company absorbs bad debt. If low-risk users are flagged as high-risk, the business loses strong customers and revenue.
That means the goal is not to build a model that merely looks clever. The goal is to support better tradeoffs between growth and risk.

Translated into machine learning terms:

| Business concept | ML counterpart |
| --- | --- |
| One user | One sample |
| User profile, billing behavior, delinquency history | Features |
| Whether the user defaults within 90 days | Label |
| Lower bad debt, better approval efficiency | Business objective |
| AUC, Recall, bad debt rate, approval rate | Evaluation outcomes |

Key idea: the first step in machine learning is not model selection. It is problem definition.

## 2. What Kinds of ML Tasks Exist

### 2.1 Supervised Learning

Supervised learning is for problems where you already have historical outcomes.

- Regression: predict a continuous value such as a risk score or expected overdue amount
- Classification: predict a discrete outcome such as default or fraud

### 2.2 Unsupervised Learning

Unsupervised learning is for problems without ground-truth labels where you still want to uncover structure.

- Clustering: group similar users to support risk segmentation
- Dimensionality reduction: compress high-dimensional features for visualization or denoising

### 2.3 Common Task Boundaries in Credit Risk

- Credit approval is usually a classification problem
- Credit limit assignment is usually a regression or ranking problem
- Discovering unusual user groups often uses clustering or anomaly detection

Task type determines the label shape, loss function, and evaluation strategy. Those decisions cannot be mixed casually.

## 3. Four Core Supervised Methods

| Method | Core intuition | Good fit | Common weakness |
| --- | --- | --- | --- |
| Linear regression | Fit a linear function to a continuous target | Risk score or limit prediction | Limited expressiveness |
| Logistic regression | Learn the probability of the positive class | Binary classification baselines, scorecards | Needs manual features for complex nonlinearity |
| Decision tree | Split the feature space with if-else rules | Rule-heavy settings with interpretability needs | Easy to overfit |
| SVM | Find the maximum-margin boundary between classes | Medium-scale classification with clear separation | Expensive at larger scale |

### Why Logistic Regression Remains a Strong Credit-Risk Baseline

Logistic regression is not “outdated.” It has survived in credit modeling because it provides several practical advantages at once:

- it outputs probabilities, which makes threshold-based decisions natural,
- coefficient signs are interpretable, which helps communication with business and compliance teams,
- it is stable on structured tabular data,
- with good feature engineering, it often produces a very competitive baseline.

Its core form is:

$$
P(y=1|x)=\sigma(w^\top x + b)=\frac{1}{1+e^{-(w^\top x+b)}}
$$

The model first computes a linear score, then converts it into a default probability between 0 and 1.

## 4. Unsupervised Learning Still Matters

### 4.1 K-Means Clustering

K-Means groups users so that users within the same cluster are as similar as possible.

In credit risk, it can help with:

- discovering behavioral segments,
- supporting risk personas,
- building layered rules for different customer groups.

### 4.2 PCA

When you have many correlated features, PCA can compress them into a smaller number of principal components.

That helps with:

- visualization,
- understanding major directions of variation,
- reducing noise in some settings.

Unsupervised learning is usually an auxiliary tool. It reveals structure, but it rarely replaces the final supervised model.

## 5. Math and Modeling Intuition

### 5.1 What Linear Regression Optimizes

Linear regression minimizes squared prediction error:

$$
\text{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Large prediction mistakes create large loss, so the optimizer adjusts $w,b$ to reduce the overall error.

### 5.2 Why Logistic Regression Outputs a Probability

Logistic regression does not directly output a class. It first computes a score, then uses a sigmoid to map that score into a probability.
That makes it especially useful in settings where downstream decisions are threshold-based.

### 5.3 What Gradient Descent Is Doing

The update rule is:

$$
\theta = \theta - \alpha \nabla J(\theta)
$$

The intuition is simple: move one small step in the direction that decreases the loss fastest.
The same idea will still hold in deep learning, just with many more parameters and deeper computational paths.

## 6. Evaluation and Generalization: Why Accuracy Can Mislead

### 6.1 Train, Validation, and Test

- Train set: learn parameters
- Validation set: tune hyperparameters and compare models
- Test set: perform the final independent evaluation

In credit risk, time-based splits are often more realistic than random splits because production always predicts the future from the past.

### 6.2 Why AUC and Recall Matter More Than Accuracy

If only 5% of users default, a model that predicts “no default” for everyone can still reach 95% accuracy while being useless in practice.

More relevant metrics are:

| Metric | What it emphasizes | Meaning in credit risk |
| --- | --- | --- |
| Recall | How many true risky users you catch | Miss fewer bad borrowers |
| Precision | How many flagged users are actually risky | Hurt fewer good borrowers |
| AUC | Overall ranking quality | Measure separation between low and high risk |
| F1 | Balance between Precision and Recall | Useful under class imbalance |

In credit risk, the key question is often whether risky users can be identified early enough, not how many examples were correct on average.

### 6.3 Bias, Variance, and Overfitting

- High bias: the model is too simple to capture the pattern
- High variance: the model is too complex and memorizes noise

Cross-validation, regularization, and feature selection are all ways to manage that tradeoff.

## 7. Feature Engineering and Data Problems

On structured data, feature engineering is often more important than switching models.

Common feature groups include:

- user profile attributes: age, occupation, income,
- credit information: limit, utilization, application frequency,
- behavioral features: repayment history, delinquency counts, spending changes,
- derived features: debt-to-income ratio, utilization change, max consecutive overdue days.

Common data problems:

- missing values: absence itself may carry risk signal,
- outliers: extreme usage may distort the distribution,
- class imbalance: risky users are much rarer,
- leakage: future information appears in training features.

### Why Leakage Inflates Offline Results

If you use a “repayment record 30 days after loan approval” to predict “default within the next 90 days,” the model will look excellent because you already slipped future information into the input.
That is not intelligence. It is an experimental mistake.

Any information unavailable at decision time should never appear in training features.

## 8. Running Case Study: Logistic Regression Baseline

### 8.1 Example Fields

| Field | Meaning |
| --- | --- |
| `age` | User age |
| `monthly_income` | Monthly income |
| `credit_limit` | Current credit line |
| `utilization_30d` | Credit utilization over the last 30 days |
| `late_payments_90d` | Number of late payments in the last 90 days |
| `bill_growth_3m` | Billing growth rate over the last three months |
| `is_default_90d` | Whether the user defaults in the next 90 days |

### 8.2 Minimal Baseline Code

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = [
    "age",
    "monthly_income",
    "credit_limit",
    "utilization_30d",
    "late_payments_90d",
    "bill_growth_3m",
]
categorical_features = ["employment_type", "city_tier"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
    ]
)

model = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=500, class_weight="balanced")),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.35).astype(int)

print("AUC:", roc_auc_score(y_test, probs))
print(classification_report(y_test, preds, digits=4))
```

### 8.3 What Error Analysis Should Look Like

Suppose the model has a decent AUC, but many missed defaults come from users with stable-looking income whose recent billing growth and utilization suddenly spiked. That suggests the model is not capturing change dynamics strongly enough. In business terms, those misses become direct bad-debt exposure.

On the other hand, if false positives are concentrated among long-term good users with only one accidental late payment, the model may be overreacting to short-term noise. In business terms, approval rate drops and strong customers get penalized.

Error analysis is not optional. It is what tells you what to fix next in features, thresholds, and model choice.

## 9. Common Mistakes

- Mistake 1: High accuracy means the model is good.
  Fix: under class imbalance, recall, AUC, and business loss matter more.
- Mistake 2: A more complex model is automatically better.
  Fix: a strong baseline is the reference point for every future iteration.
- Mistake 3: Random splitting is always fine.
  Fix: for time-dependent data, random splitting can create false optimism.
- Mistake 4: More features always help.
  Fix: redundant, noisy, and leaked features all hurt reliability.

## 10. What You Should Remember

- Machine learning begins with problem definition, not model worship.
- Logistic regression is still a strong baseline for structured credit-risk tasks because it is stable, interpretable, and deployable.
- Metrics must be read together with business cost, especially under class imbalance.
- In structured-data projects, splitting strategy, feature engineering, and error analysis often matter more than changing the model.

## Next Step

If this chapter taught you how to turn a business request into a credible ML task, the next chapter answers a new question:
when manual features stop being enough and linear boundaries start to fail, how can a model learn richer representations automatically?

Move on to [Deep Learning Basics](../deep-learning-basics/README_EN.md).
