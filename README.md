# Online Retail

### Online Retail dataset performing customer segmentation using Python and focusing on clustering and classification techniques.

Just as an example, for a small company, the cutomer flow is small and they are targetable. But in the business case, if business grows in size, it will not be possible for the business to target about each and every customer as the customer size also grows. It is important not just to invest on acquiring new customers but also in customer retention. The business get most of the revenue from the old customers as well as high-valued and loyal customers. As the company resources are limited, to find the customers needs and target them. So it is also important to find the customers whose retention is less i.e who are at high-risk.
Customer Segmentation is the process of dividing customers into groups on the basis of attritubtes or behavior. Insights from Customer Segmentation helps comapny to understand during the marketing campaigns and also to plan marketing strategy. So used the regression and classification techniques to get the insights from the data. While using Lasso and Ridge Regression technique haven't much got the insights from the data. So used some other like DecisionTreeClassifier, SVM, RandomForestClassifier by analyzing the accuracy score, DecisioTreeClassifier model has good accuracy score.To understand better worked on clustering technique i.e KMeans clustering to get more insights from the data.
	
### Conclusion: 
As can be seen in the below boxplot cluster0 and cluster1 customers spend more money but buy less products where as cluster3 spend less amount then cluster1 and cluster1 but buy more products. As in cluster2 the amount spend on products and the quantity bought are more they can be our loyal customers. These are the insights from the dataset. 


![total_amount](https://user-images.githubusercontent.com/67755812/205316394-87a38a10-dba0-4e31-9fae-c2535740d98f.png)



![quantity](https://user-images.githubusercontent.com/67755812/205316417-30b981c5-a789-4861-a78b-06f94c1d1d60.png)






For more insights have also done with rfm technique. You can also check that.


## RFM Analysis:
RFM Analysis is a technique used for customer segmentation. It enables customers to be divided into groups based on their purchasing habits and to develop strategies specific to these groups.

Metrics: R: Recency (innovation): Time from customer's last purchase to date. F: Frequency: Total number of purchases. M: Monetary: Customer's total expenditure.

Hibernating -> R:[1-2], F:[1-2]
At Risk -> R:[1-2], F:[3-4]
Can't lose -> R:[1-2], F:5
About to Sleep -> R:3 , F:[1-2]
Need Attention -> R:3 , F:3
Loyal Customers -> R:[3-4], F:[4-5]
Promising -> R:4, F: 1
New_Customers -> R:5, F: 1
Potential Loyalists -> R:[4-5], F:[2-3]
Champions-> R:5, F:[4-5]

### Conclusion: 
There are 594 customers in the AtRisk group. The frequency of the group's purchase mean is 3.1. There are 66 new customers. Their last purchase was 2 days ago. There are 91 customers in can't loose group. They can be back with marketing campaigns.


![rfm](https://user-images.githubusercontent.com/67755812/205316465-4f8a7bb7-0da9-43f1-bc5c-ebd8cd7d17c8.PNG)

