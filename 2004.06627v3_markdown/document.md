# An Application of Deep Reinforcement Learning to Algorithmic Trading

Thibaut Th ́eatea,∗, Damien Ernsta

a Montefiore Institute, University of Li`ege (All ́ee de la d ́ecouverte 10, 4000 Li`ege, Belgium)

Abstract

This scientific research paper presents an innovative approach based on deep reinforcement learning (DRL) to solve the algorithmic trading problem of determining the optimal trading position at any point in time during a trading activity in the stock market. It proposes a novel DRL trading policy so as to maximise the resulting Sharpe ratio performance indicator on a broad range of stock markets. Denominated the Trading Deep Q-Network algorithm (TDQN), this new DRL approach is inspired from the popular DQN algorithm and significantly adapted to the specific algorithmic trading problem at hand. The training of the resulting reinforcement learning (RL) agent is entirely based on the generation of artificial trajectories from a limited set of stock market historical data. In order to objectively assess the performance of trading strategies, the research paper also proposes a novel, more rigorous performance assessment methodology. Following this new performance assessment approach, promising results are reported for the TDQN algorithm.

Keywords: Artificial intelligence, deep reinforcement learning, algorithmic trading, trading policy.

1. Introduction

For the past few years, the interest in artificial intellicluding the problems related to trading, investment, risk gence (AI) has grown at a very fast pace, with numerous management, portfolio management, fraud detection and research papers published every year. A key element for financial advising, to cite a few. Such complex decision-this growing interest is related to the impressive successes making problems are extremely complex to solve as they ofdeep learning (DL) techniques which are based on deep generally have a sequential nature and are highly stochas-neural networks (DNN) \- mathematical models directly intic, with an environment partially observable and poten-spired by the human brain structure. These specific techtially adversarial. In particular, algorithmic trading, which niques are nowadays the state of the art in many appliis a key sector of the Fin Tech industry, presents particu-cations such as speech recognition, image classification or larly interesting challenges. Also called quantitative trad-natural language processing. In parallel to DL, another ing, algorithmic trading is the methodology to trade using field of research has recently gained much more attention computers and a specific set of mathematical rules. ing (DRL). This family of techniques is concerned with The main objective of this research paper is to an-the learning process of an intelligent agent (i) interacting swer the following question: how to design a novel trad-in a sequential manner with an unknown environment (ii) ing policy (algorithm) based on AI techniques that could aiming to maximise its cumulative rewards and (iii) uscompete with the popular algorithmic trading strategies ing DL techniques to generalise the information acquired widely adopted in practice? To answer this question, this from the interaction with the environment. The many rescientific article presents and analyses a novel DRL solu-cent successes of DRL techniques highlight their ability to tion to tackle the algorithmic trading problem of deter-solve complex sequential decision-making problems. mining the optimal trading position (long or short) at any Nowadays, an emerging industry which is growing ex The algorithmic solution presented in this research paper tremely fast is the financial technology industry, generally is inspired by the popular Deep Q-Network (DQN) algo-referred to by the abbreviation Fin Tech. The objective of rithm, which has been adapted to the particular sequential Fin Tech is pretty simple: to extensively take advantage decision-making problem at hand. The research question of technology in order to innovate and improve activities to be answered is all the more relevant as the trading envi-in finance. In the coming years, the Fin Tech industry is ronment presents very different characteristics from those

### from the research community: deep reinforcement learn-

> ∗Corresponding author.
> Email addresses: thibaut.theate@uliege.be (Thibaut
> Th ́eate), dernst@uliege.be (Damien Ernst)

```text
expected to revolutionise the way many decision-making
problems related to the financial sector are addressed, in-
point in time during a trading activity in the stock market.
```

which have already been successfully solved by DRL ap-proaches, mainly significant stochasticity and extremely poor observability.

---

The scientific research paper is structured as follows. et al. (2017) which introduced the fuzzy recurrent deep First of all, a brief review of the scientific literature around neural network structure to obtain a technical-indicator-the algorithmic trading field and its main AI-based contrifree trading system taking advantage of fuzzy learning to butions is presented in Section 2. Afterwards, Section 3 inreduce the time series uncertainty. One can also mention troduces and rigorously formalises the particular algorith Carapu ̧co et al. (2018) which studied the application of the mic trading problem considered. Additionally, this section deep Q-learning algorithm for trading in foreign exchange makes the link with the reinforcement learning (RL) apmarkets. Finally, there exist a few interesting works study-proach. Then, Section 4 covers the complete design of the ing the application of DRL techniques to algorithmic trad-TDQN trading strategy based on DRL concepts. Subseing in specific markets, such as in the field of energy, see 6 is concerned with the presentation and discussion of the To finish with this short literature review, a sensi-results achieved by the TDQN trading strategy. To end tive problem in the scientific literature is the tendency to this research paper, Section 7 discusses interesting leads prioritise the communication of good results or findings, as future work and draws meaningful conclusions. sometimes at the cost of a proper scientific approach with

quently, Section 5 proposes a novel methodology to objece.g. the article Boukas et al. (2020).
tively assess the performance of trading strategies. Section

2. Literature review

To begin this brief literature review, two facts have to pears to be all the more relevant in the field of financial be emphasised. Firstly, it is important to be aware that sciences, especially when the subject directly relates to many sound scientific works in the field of algorithmic tradtrading activities. Indeed, Bailey et al. (2014) claims that ing are not publicly available. As explained in Li (2017), many scientific publications in finance suffer from a lack due to the huge amount of money at stake, private Fin Tech of a proper scientific approach, instead getting closer to firms are very unlikely to make their latest research results pseudo-mathematics and financial charlatanism than rig-public. Secondly, it should be acknowledged that making a orous sciences. Aware of these concerning tendencies, the fair comparison between trading strategies is a challenging present research paper intends to deliver an unbiased sci-task, due to the lack of a common, well-established frameentific evaluation of the novel DRL algorithm proposed. trading costs which are variously defined or even omitted. In this section, the sequential decision-making algorith-First of all, most of the works in algorithmic trading are sented in detail. Moreover, a rigorous formalisation of this techniques developed by mathematicians, economists and particular problem is performed. Additionally, the link (2009), Chan (2013) and Narang (2009). Then, the major Algorithmic trading, also called quantitative trading, ity of works applying machine learning (ML) techniques in is a subfield of finance, which can be viewed as the ap-the algorithmic trading field focus on forecasting. If the proach of automatically making trading decisions based on financial market evolution is known in advance with a reaa set of mathematical rules computed by a machine. This sonable level of confidence, the optimal trading decisions commonly accepted definition is adopted in this research can easily be computed. Following this approach, DL techpaper, although other definitions exist in the literature. niques have already been investigated with good results, Indeed, several authors differentiate the trading decisions see e.g. Ar ́evalo et al. (2016) introducing a trading strat (quantitative trading) from the actual trading execution egy based on a DNN, and especially Bao et al. (2017) using (algorithmic trading). For the sake of generality, algo-wavelet transforms, stacked autoencoders and long shortrithmic trading and quantitative trading are considered term memory (LSTM). Alternatively, several authors have synonyms in this research paper, defining the entire auto-already investigated RL techniques to solve this algorithmated trading process. Algorithmic trading has already mic trading problem. For instance, Moody and Saffell proven to be very beneficial to markets, the main benefit (2001) introduced a recurrent RL algorithm for discoverbeing the significant improvement in liquidity, as discussed ing new investment policies without the need to build forein Hendershott et al. (2011). For more information about casting models, and Dempster and Leemans (2006) used this specific field, please refer to Treleaven et al. (2013) entifically sound way to solve this particular algorithmic There are many different markets suitable to apply al-trading problem. For instance, one can first mention Deng gorithmic trading strategies. Stocks and shares can be

work to properly evaluate their performance. Instead, the authors generally define their own framework with their evident bias. Another major problem is related to the

traders who do not exploit AI. Typical examples of claswith the RL formalism is highlighted. sical trading strategies are the trend following and mean reversion strategies, which are covered in detail in Chan

adaptive RL to trade in foreign exchange markets. More and Nuti et al. (2011).
recently, a few works investigated DRL techniques in a sci-

```text
objective criticism. Going even further, Ioannidis (2005)
even states that most published research findings in cer-
tain sensitive fields are probably false. Such concern ap-
```

3. Algorithmic trading problem formalisation

### mic trading problem studied in this research paper is pre-

### 3.1. Algorithmic trading

2

---

```text
traded in the stock markets, FOREX trading is concerned The duration ∆tis closely linked to the trading fre-
kets planned in the future.
```

with foreign currencies, or a trader could invest in comquency targeted by the trading agent (very high trading modity futures, to only cite a few. The recent rise of frequency, intraday, daily, monthly, etc.). Such discretisa-cryptocurrencies, such as the Bitcoin, offers new intertion operation inevitably imposes a constraint with respect esting possibilities as well. Ideally, the DRL algorithms to this trading frequency. Indeed, because the duration ∆t developed in this research paper should be applicable to between two time steps cannot be chosen as small as pos-multiple markets. However, the focus will be set on stock sible due to technical constraints, the maximum trading markets for now, with an extension to various other marfrequency achievable, equal to 1/∆t, is limited. In the the scope of this research paper, the portfolio considered The algorithmic trading approach is rule based, mean-consists of one single stock together with the agent cash. ing that the trading decisions are made according to a set The portfolio valuevtis then composed of the trading of rules: a trading strategy. In technical terms, a trading t t strategy can be viewed as a programmed policyπ (at|it), uously evolves over timet. Buying and selling operations either deterministic or stochastic, which outputs a trad-are simply cash and share exchanges. The trading agent ing actionaaccording to the information available to the interacts with the stock market through an order book, trading agentitat time stept. Additionally, a key char-which contains the entire set of buying orders (bids) and acteristic of a trading strategy is its sequential aspect, as selling orders (asks). An example of a simple order book illustrated in Figure 1. An agent executing its trading

In fact, a trading activity can be viewed as the maning agent makes a new decision once every day. agement of a portfolio, which is a set of assets including diverse stocks, bonds, commodities, currencies, etc. In 3.3. Trading strategy agent cash valuevcand the share valuevs, which contin-

is depicted in Table 1. An order represents the willingness strategy sequentially applies the following steps: of a market participant to trade and is composed of a price p, a quantityqand a sides (bid or ask). For a trade to occur, a match between bid and ask orders is required, an 2. Execution of the policyπ (at|it) to get actionat. event which can only happen ifpbid ≥pask, withpbid 3. Application of the designated trading actiona.

### max min max

```text
(pask) being the maximum (minimum) price of a bid (ask) 4. Next time stept→t+ 1, loop back to step 1.
min
```

order. Then, a trading agent faces a very difficult task in order to generate profit: what, when, how, at which price and which quantity to trade. This is the algorithmic trad-ing complex sequential decision-making problem studied in this scientific research paper.

```text
Table 1: Example of a simple order book
Sides Quantityq Pricep
Ask 3000 107 community, is casted as an RL problem.
Ask 1500 106
Ask 500 105
Bid 1000 95
Bid 2000 94
Bid 4000 93
```

### 3.2. Timeline discretisation

Since trading decisions can be issued at any time, the resulting from its RL policyπ (at|ht) wherehtis the RL trading activity is a continuous process. In order to study agent history and receives a rewardrtas a consequence of the algorithmic trading problem described in this research its action. In this RL context, the agent history can be

![Page 3 image](images/page-003-image-01.png)

paper, a discretisation operation of the continuous timeexpressed asht={(oτ, aτ, rτ)|τ= 0, 1, ..., t}. line is performed. The trading timeline is discretized into a high number of discrete trading time stepstof constant duration ∆t. In this research paper, for the sake of clarity, the increment (decrement) operationst+ 1 (t−1) are used to model the discrete transition from time steptto time stept+ ∆t (t−∆t).

scope of this research paper, this constraint is met as the trading frequency targeted is daily, meaning that the trad-

*t*

1. Update of the available market informationit.

*t*

### Figure 1: Illustration of a trading strategy execution

In the following subsection, the algorithmic trading se-quential decision-making problem, which shares similari-ties with other problems successfully tackled by the RL 3.4. Reinforcement learning problem formalisation As illustrated in Figure 2, reinforcement learning is concerned with the sequential interaction of an agent with its environment. At each time stept, the RL agent firstly observes the RL environment of internal statest, and re-trieves an observationot. It then executes the actionat

```text
Reinforcement learning techniques are concerned with
the design of policiesπmaximising an optimality crite-
rion, which directly depends on the immediate rewardsrt
observed over a certain time horizon. The most popular
```

3

---

optimality criterion is the expected discounted sum of re At each trading time stept, the RL agent observes resulting optimal policyπ∗is expressed as the following: limited information collected by the agent on this complex

wards over an infinite time horizon. Mathematically, the the stock market whose internal state isst∈ S.The

```text
π= argmax E[R|π] (1) observation space Oshould encompass all the information
```

### ∗

> π
> ∑∞
> t=0

```text
R= γrt (2) observationothas to be considered as a sequence of both
```

*t*

The parameterγis the discount factor (γ∈[0, 1]). It determines the importance of future rewards. For in-stance, ifγ= 0, the RL agent is said to be myopic as it only considers the current reward and totally discards the future rewards. When the discount factor increases, the RL agent tends to become more long-term oriented. In the extreme case whereγ= 1, the RL agent considers each reward equally. This key parameter should be tuned according to the desired behaviour.

### Figure 2: Reinforcement learning core building blocks

3.4.1. RL observations In the scope of this algorithmic trading problem, the RL environment is the entire complex trading world grav-itating around the RL agent. In fact, this trading envi-ronment can be viewed as an abstraction including the trading mechanisms together with every single piece of in-formation capable of having an effect on the trading act tivity of the agent. A major challenge of the algorithmic trading problem is the extremely poor observability of this Vtis the total volume of shares exchanged over environment. Indeed, a significant amount of information is simply hidden to the trading agent, ranging from some companies’ confidential information to the other market participants’ strategies. In fact, the information available to the RL agent is extremely limited compared to the com-plexity of the environment. Moreover, this information can take various forms, both quantitative and qualitative. Correctly processing such information and re-expressing it using relevant quantitative figures while minimising the subjective bias is capital. Finally, there are significant time correlation complexities to deal with. Therefore, the infor-mation retrieved by the RL agent at each time step should than individually.

be considered sequentially as a series of information rather •M (t) gathers the macroeconomic information at the

trading environment is denoted byot∈ O.Ideally, this capable of influencing the market prices. Because of the sequential aspect of the algorithmic trading problem, an the information gathered during the previousτtime steps (history) and the newly available information at time step t. In this research paper, the RL agent observations can be mathematically expressed as the following:

### ot={S (t), D (t), T (t), I (t), M (t), N (t), E (t)}t′=t−τ

### ′ ′ ′ ′ ′ ′ ′ t

### (3)

![Page 4 image](images/page-004-image-01.png)

### where:

•S (t) represents the state information of the RL agent at time stept (current trading position, number of shares owned by the agent, available cash). •D (t) is the information gathered by the agent at time steptconcerning the OHLCV (Open-High-Low-Close-Volume) data characterising the stock market. More precisely, D (t) can be expressed as follows:

### D (t) ={pt, pt, pt, pt, Vt} (4)

### O H L C

where: p Ois the stock market price at the opening of

*t*

the time period [t−∆t, t[. p H is the highest stock market price over the

*t*

time period [t−∆t, t[. p Lis the lowest stock market price over the time

*t*

```text
period [t−∆t, t[.
p C is the stock market price at the closing of
the time period [t−∆t, t[.
```

the time period [t−∆t, t[.

•T (t) is the agent information regarding the trading time stept (date, weekday, time). •I (t) is the agent information regarding multiple tech-nical indicators about the stock market targeted at time stept. There exist many technical indicators providing extra insights about diverse financial phe-nomena, such as moving average convergence diver-gence (MACD), relative strength index (RSI) or av-erage directional index (ADX), to only cite a few.

disposal of the agent at time stept. There are many interesting macroeconomic indicators which could po-tentially be useful to forecast markets’ evolution, such as the interest rate or the exchange rate.

4

---

•N (t) represents the news information gathered by Actually, the real actions occurring in the scope of a the agent at time stept. These news data can be trading activity are the orders posted on the order book. extracted from various sources such as social media The RL agent is assumed to communicate with an external (Twitter, Facebook, Linked In), the newspapers, spemodule responsible for the synthesis of these true actions cific journals, etc. Complex sentiment analysis modaccording to the value of Qt: thetrading execution system. els could then be built to extract meaningful quanti Despite being out of the scope of this paper, it should be tative figures (quantity, sentiment polarity and submentioned that multiple execution strategies can be con-eral authors, see e.g. Leinweber and Sisk (2011), The trading actions have an impact on the two com-Bollen et al. (2011) and Nuij et al. (2014). ponents of the portfolio value, namely the cash and share

jectivity, etc.) from the news. The benefits of such sidered depending on the general trading purpose. information has already been demonstrated by sev-•E (t) is any extra useful information at the disposal of the trading agent at time stept, such as other market participants trading strategies, companies’ confiden-tial information, similar stock market behaviours, c c rumours, experts’ advice, etc.

Observation space reduction: In the scope of this research paper, it is assumed that the only information considered by the RL agent is the withnt∈Zbeing the number of shares owned by the classical OHLCV data D (t) together with the state infor-mation S (t). Especially, the reduced observation space O encompasses the current trading position together with a spite being surprising at first glance, a negative number series of the previousτ+1 daily open-high-low-close prices and daily traded volume. With such an assumption, the reduced RL observationotcan be expressed as the follow-ing:

### ot= {pt′, pt′, pt′, pt′, Vt′}t′=t−τ, Pt (5)

### {}

> O H L C t Two important constraints are assumed concerning the

with Ptbeing the trading position of the RL agent at time stept (either long orshort, as explained in the next sub-section of this research paper).

3.4.2. RL actions At each time stept, the RL agent executes a trading ac-tionat∈ Aresulting from its policyπ (at|ht). In fact, the trading agent has to answer several questions: whether, how and how much to trade? Such decisions can be mod-elled by the quantity of shares bought by the trading agent at time stept, represented by Qt∈Z. Therefore, the RL actions can be expressed as the following:

### at=Qt (6)

Three cases can occur depending on the value of Qt: •Qt<0: The RL agentsells shares on the stock mar-ket, by posting new ask orders on the order book. •Qt= 0: The RL agent holds, meaning that it does not buy nor sell any shares on the stock market.

•Qt>0: The RL agentbuys shares on the stock maras long as the market variation remains below this value. ket, by posting new bid orders on the order book. Therefore, the constraints acting upon the RL actions at

values. Assuming that the trading actions occur close to the market closure at pricept'pt, the updates of these

*C*

### components are governed by the following equations:

```text
vt+1=vt−Qtpt (7)
vt+1= (nt+Qt) pt+1 (8)
```

*s*

> ︸ ︷︷ ︸
> nt+1

trading agent at time stept. In the scope of this research paper, negative values are allowed for this quantity. De-of shares simply corresponds to shares borrowed and sold, with the obligation to repay the lender in shares in the future. Such a mechanism is particularly interesting as it introduces new possibilities for the trading agent.

quantity of traded shares Qt. Firstly, contrarily to the share valuevswhich can be both positive or negative, the

*t*

### cash valuevchas to remain positive for every trading time

*t*

stepst. This constraint imposes an upper bound on the number of shares that the trading agent is capable of pur-chasing, this volume of shares being easily derived from Equation 7. Secondly, there exists a risk associated with the impossibility to repay the share lender if the agent suffers significant losses. To prevent such a situation from happening, the cash valuevcis constrained to be suffi-

*t*

```text
ciently large when a negative number of shares is owned, in
order to be able to get back to a neutral position (nt= 0).
A maximum relative change in prices, expressed in % and
denoted ∈R, is assumed by the RL agent prior to
```

### +

the trading activity. This parameter corresponds to the maximum market daily evolution supposed by the agent over the entire trading horizon, so that the trading agent should always be capable of paying back the share lender time steptcan be mathematically expressed as follows:

### vt+1≥0 (9)

*c*

### vt+1≥ −nt+1pt (1 +) (10)

*c*

5

---

with the following condition assumed to be satisfied: is a truly complex task. In this research paper, the in-∣ ∣ ∣pt+1−pt∣ ∣ ∣≤ ∣ p ∣

*t*

### (11) formed through a heuristic. When a trade is executed, a

Trading costs consideration: Actually, the modelling represented by Equation 7 is inaccurate and will inevitably lead to unrealistic results. Indeed, whenever simulating trading activities, the trading costs should not be neglected. Such omission is generally misleading as a trading strategy, highly profitable in simu-lations, may be likely to generate large losses in real trad-ing situations due to these trading costs, especially when the trading frequency is high. The trading costs can be subdivided into two categories. On the one hand, there are explicit costs which are induced by transaction costs Moreover, the trading costs have to be properly consid-and taxes. On the other hand, there are implicit costs, ered in the constraint expressed in Equation 10. Indeed, called slippage costs, which are composed of three main the cash valuevcshould be sufficiently large to get back elements and are associated to some of the dynamics of to a neutral position (nt= 0) when the maximum mar-the trading environment. The different slippage costs are ket variation occurs, the trading costs being included. detailed hereafter: •Spread costs: These costs are related to the differc ence between the minimum ask pricepask and the

*min*

### maximum bid pricepbid, called the spread. Because

*max*

```text
the complete state of the order book is generally too discrete set of acceptable values for the quantity of traded
complex to efficiently process or even not available, shares Qt. Derived in detail in Appendix A, the RL action
the trading decisions are mostly based on the midspace Ais mathematically expressed as the following:
dle pricepmid= (pbid +pask)/2. However, a buying
```

### max min

> (selling) trade issued atpmid inevitably occurs at a
> pricep≥pmin (p≤pmax). Such costs are all the

### ask bid

more significant that the stock market liquidity is low compared to the volume of shares traded. •Market impact costs: These costs are induced by the impact of the trader’ s actions on the market. •Qt= Each trade (both buying and selling orders) is popt (2C+(1+C)) tentially capable of influencing the price. This phewith ∆t=−vt−ntpt (1 +)(1 +C). nomenon is all the more important that the stock market liquidity is low with respect to the volume of shares traded. •Timing costs: These costs are related to the time required for a trade to physically happen once the trading decision is made, knowing that the market price is continuously evolving. The first cause is the inevitable latency which delays the posting of the orders on the market order book. The second cause is the intentional delays generated by the trading execution system. For instance, a large trade could be split into multiple smaller trades spread over time shares owned by the trading agent, by converting as much in order to limit the market impact costs. An accurate modelling of the trading costs is required to realistically reproduce the dynamics of the real trad-ing environment. While explicit costs are relatively easy t to take into account, the valid modelling of slippage costs

tegration of both costs into the RL environment is per-certain amount of capital equivalent to a percentage Cof the amount of money invested is lost. This parameter was realistically chosen equal to 0.1% in the forthcoming sim-ulations.

```text
Practically, these trading costs are directly withdrawn
from the trading agent cash. Following the heuristic pre-
viously introduced, Equations 7 can be re-expressed with
a corrective term modelling the trading costs:
```

> vt+1=vt−Qtpt−C|Qt|pt (12)
> c c

> ︸ ︷︷ ︸
> Trading costs

*t*

### Consequently, Equation 10 is re-expressed as follows:

```text
vt+1≥ −nt+1pt (1 +)(1 +C) (13)
Eventually, the RL action space Acan be defined as the
```

### A={Qt∈Z∩[Qt, Qt]} (14)

*where:
•Qt=*

> vt
> c

> pt (1+C)
> {

> ∆t
> pt (1+C)

### if ∆t≥0

### ∆t

### if ∆t<0

*c*

> Action space reduction:
> In the scope of this scientific research paper, the ac-
> tion space Ais reduced in order to lower the complexity
> of the algorithmic trading problem. The reduced action
> space is composed of only two RL actions which can be
> mathematically expressed as the following:

### at=Qt∈ {Qt, Qt} (15)

### Long Short

### The first RL action Qt maximises the number of

### Long

### cash valuevcas possible into share valuevs. It can be

### t t

### mathematically expressed as follows:

> QLong= pt (1+C)
> {⌊ c

### vt Long

> ⌋
> ifat−16=Qt−1,

0 otherwise.

### (16)

6

---

The action Qt is always valid as it is obviously inis particularly well suited for the performance assessment

### Long

```text
cluded into the original action space Adefined by Equatask as it considers both the generated profit and the risk
tion 14. As a result of this action, the trading agent owns associated with the trading activity. Mathematically, the
a number of shares Nt =nt+Qt. On the contrary, Sharpe ratio Sris expressed as the following:
```

### Long Long

### the second RL action, designated by QShort, converts share

*t*

### valuevsinto cash valuevc, such that the RL agent owns

### t t

```text
a number of shares equal to−NLong. This operation can Sr= =√ '√ (20)
```

*t*

### be mathematically expressed as the following:

> Q̂Short= pt (1+C)
> t

> {
> −2nt− ifat−16=Qt−1,

> ⌊ ⌋
> vt Short

*c*

```text
0 otherwise.
```

(17) period, modelling its profitability.

### However, the action Q̂Shortmay violate the lower bound

*t*

Qtof the action space Awhen the price significantly in•σris the standard deviation of the trading strategy creases over time. Eventually, the second RL action QShort excess return Rs−Rf, modelling its riskiness.

*t*

### is expressed as follows:

> QShort= max Q̂Short, Q (18)
> t t

### {}

*t*

```text
To conclude this subsection, it should be mentioned the ratio between the returns mean and standard devia-
that the two reduced RL actions are actually related to tion is evaluated. Finally, the annualised Sharpe ratio is
the next trading position of the agent, designated as Pt+1.
Indeed, the first action Qt induces a long trading po-
```

### Long

sition because the number of owned shares is positive. On the contrary, the second action QShort always results in

*t*

a number of shares which is negative, which is generally ideally be capable of achieving acceptable performance on referred to as a short trading position in finance.

### 3.4.3. RL rewards

For this algorithmic trading problem, a natural choice decreasing price trends), with different levels of volatility. for the RL rewards is the strategy daily returns. Intu-itively, it makes sense to favour positive returns which are opment of a novel trading strategy based on DRL tech-an evidence of a profitable strategy. Moreover, such quanniques to maximise the average Sharpe ratio computed on tity has the advantage of being independent of the number the entire set of existing stock markets. of sharesntcurrently owned by the agent. This choice is also motivated by the fact that it allows to avoid a sparse reward setup, which is more complex to deal with. The RL imisation of the Sharpe ratio, the DRL algorithm adopted rewards can be mathematically expressed as the following: in this scientific paper actually maximises the expected

### rt= (19)

### vt+1−vt

vt exactly corresponds to maximising profits but is very close

### 3.5. Objective

Objectively assessing the performance of a trading stratbe to narrow the gap between these two objectives. egy is a tricky task, due to the numerous quantitative and qualitative factors to consider. Indeed, a well-performing trading strategy is not simply expected to generate profit, but also to efficiently mitigate the risk associated with In this section, a novel DRL algorithm is designed to the trading activity. The balance between these two goals solve the algorithmic trading problem previously intro-varies depending on the trading agent profile and its willduced. The resulting trading strategy, denominated the ingness to take extra risks. Although intuitively conve Trading Deep Q-Network algorithm (TDQN), is inspired nient, maximising the profit generated by a trading stratfrom the successful DQN algorithm presented in Mnih egy is a necessary but not sufficient objective. Instead, et al. (2013) and is significantly adapted to the specific the core objective of a trading strategy is the maximisadecision-making problem at hand. Concerning the train-tion of the Sharpe ratio, a performance indicator widely ing of the RL agent, artificial trajectories are generated used in the fields of finance and algorithmic trading. It from a limited set of stock market historical data.

> E[Rs−Rf] E[Rs−Rf] E[Rs]
> σr var[Rs−Rf] var[Rs]

### where:

•Rsis the trading strategy return over a certain time •Rfis the risk-free return, the expected return from a totally safe investment (negligible).

In order to compute the Sharpe ratio Srin practice, the daily returns achieved by the trading strategy are firstly computed using the formulaρt= (vt−vt−1)/vt−1. Then, obtained by multiplying this value by the square root of the number of trading days in a year (252).

Moreover, a well-performing trading strategy should diverse markets presenting very different patterns. For in-stance, the trading strategy should properly handle both bull and bear markets (respectively strong increasing and Therefore, the research paper’ s core objective is the devel-

### Despite the fact that the ultimate objective is the max-

discounted sum of rewards (daily returns) over an infinite time horizon. This optimisation criterion, which does not to that, can in fact be seen as a relaxation of the Sharpe ra-tio criterion. A future interesting research direction would

4. Deep reinforcement learning algorithm design

7

---

### Figure 3: Illustration of the DQN algorithm

### 4.1. Deep Q-Network algorithm

```text
The Deep Q-Network algorithm, generally referred to generated are simply composed of the sequence of histori-
as DQN, is a DRL algorithm capable of successfully learncal real observations associated with various sequences of
rithm introduced in Watkins and Dayan (1992). This DRL the trading agent should not be able to influence the stock
model of the environment is not required and that trajecthe number of shares traded by the trading agent is low
```

ing control policies from high-dimensional sensory inputs. trading actions from the RL agent. For such practice to be It is in a way the successor of the popular Q-learning algoscientifically acceptable and lead to realistic simulations, algorithm is said to bemodel-free, meaning that a complete tories are sufficient. Belonging to the Q-learning family of algorithms, it is based on the learning of an approximation of the state-action value function, which is represented by a learning the parametersθof this DNN. Finally, the DQN algorithm is said to be off-policy as it exploits in batch mode previous experienceset= (st, at, rt, st+1) collected t t at any point during training.

DNN. In such context, learning the Q-function amounts to just described, a trick is employed to slightly improve the

For the sake of brevity, the DQN algorithm is illuscopy of this environment E−. Although this trick does not

trated in Figure 3, but is not extensively presented in completely solve the challenging exploration/exploitation
this paper. Besides the original publications (Mnih et al. trade-off, it enables the RL agent to continuously explore
(2013) and Mnih et al. (2015)), there exists a great sciat a small extra computational cost.
entific literature around this algorithm, see for instance
van Hasselt et al. (2015), Wang et al. (2015), Schaul et al. 4.3. Diverse modifications and improvements
(2016), Bellemare et al. (2017), Fortunato et al. (2018)
and Hessel et al. (2017). Concerning DL techniques, interthe novel DRL trading strategy developed, but was signifi-
and surveys: Sutton and Barto (2018), Szepesvari (2010), simulations performed, are summarised hereafter:
Busoniu et al. (2010), Arulkumaran et al. (2017) and Shao
et al. (2019).

esting resources are Le Cun et al. (2015), Goodfellow et al. cantly adapted to the specific algorithmic trading decision-
(2015) and Goodfellow et al. (2016). For more information making problem at hand. The diverse modifications and
about RL, the reader can refer to the following textbooks improvements, which are mainly based on the numerous

### 4.2. Artificial trajectories generation

```text
In the scope of the algorithmic trading problem, a comnature of the input (time-series instead of raw im-
plete model of the environment Eis not available. The ages), the convolutional neural network (CNN) has
training of the TDQN algorithm is entirely based on the been replaced by a classical feedforward DNN with
```

![Page 8 image](images/page-008-image-01.png)

generation of artificial trajectories from a limited set of some leaky rectified linear unit (Leaky Re LU) acti-stock market historical daily OHLCV data. A trajectory vation functions.

```text
τis defined as a sequence of observationsot∈ O, actions
at∈ Aand rewardsrtfrom an RL agent for a certain
number Tof trading time stepst:
```

```text
τ= {o0, a0, r0},{o1, a1, r1}, ...,{o T, a T, r T}
()
```

Initially, although the environment Eis unknown, one disposes of a single real trajectory, corresponding to the historical behaviour of the stock market, i.e. the particular case of the RL agent being inactive. This original trajec-tory is composed of the historical prices and volumes to-gether with long actions executed by the RL agent with no money at its disposal, to represent the fact that no shares are actually traded. For this algorithmic trading problem, new fictive trajectories are then artificially generated from this unique true trajectory to simulate interactions with the environment E. The historical stock market behaviour is simply considered unaffected by the new actions per-formed by the trading agent. The artificial trajectories market behaviour. This assumption generally holds when with respect to the liquidity of the stock market.

In addition to the generation of artificial trajectories exploration of the RL agent. It relies on the fact that the reduced action space Ais composed of only two actions: long (Q) and short (Q). At each trading time

### Long Short

stept, the chosen actionatis executed on the trading environment Eand the opposite actionatis executed on a

### −

The DQN algorithm was chosen as starting point for •Deep neural network architecture: The first dif-ference with respect to the classical DQN algorithm is the architecture of the DNN approximating the action-value function Q (s, a). Due to the different

8

---

•Double DQN: The DQN algorithm suffers from •Xavier initialisation: While the classical DQN algorithm which is based on the decomposition of the remains constant across the DNN layers. target max operation into both action selection and action evaluation.

substantial overestimations, this overoptimism harmalgorithm simply initialises the DNN weights ran-ing the algorithm performance. In order to reduce domly, the Xavier initialisation is implemented to the impact of this undesired phenomenon, the article improve the algorithm convergence. The idea is to van Hasselt et al. (2015) presents the double DQN set the initial weights so that the gradients variance implements the RMSProp optimiser. However, the the activation functions. It brings many benefits in-ADAM optimiser, introduced in Kingma and Ba (2015), cluding a faster and more robust training phase as plements a mean squared error (MSE) loss, the Hularisation techniques are implemented: Dropout, L2 nique is implemented in the TDQN algorithm to hinted, this procedure is all the more critical because there solve the gradient exploding problem which induces has been a real lack of a proper performance assessment

•ADAM optimiser: The classical DQN algorithm normalising the input layer by adjusting and scaling experimentally proves to improve both the training well as an improved generalisation. stability and the convergence speed of the DRL al-gorithm.

•Huber loss: While the classical DQN algorithm imiments with the DRL trading strategy, three regu-ber loss experimentally improves the stability of the regularisation and Early Stopping. training phase. Such observation is explained by the fact that the MSE loss significantly penalises large errors, which is generally desired but has a negative side-effect for the DQN algorithm because the DNN is supposed to predict values that depend on its own input. This DNN should not radically change in a single training update because this would also lead to a significant change in the target, which could ac-tually result in a larger error. Ideally, the update of the DNN should be performed in a slower and more stable manner. On the other hand, the mean ab-solute error (MAE) has the drawback of not being differentiable at 0. A good trade-off between these two losses is the Huber loss H:

### H (x) = 1 (21)

> {1x2 if|x| ≤1,
> 2
> |x| − otherwise.

2

![Page 9 image](images/page-009-image-01.png)

### Figure 4: Comparison of the MSE, MAE and Huber losses

•Gradient clipping: The gradient clipping techtal in order to produce meaningful results. As previously significant instabilities during the training of the DNN. methodology in the algorithmic trading field. In this sec-

•Batch normalisation layers: This DL technique, introduced by Ioffe and Szegedy (2015), consists in •Regularisation techniques: Because a strong ten-dency to overfit was observed during the first exper-•Preprocessing and normalisation: The training loop of the TDQN algorithm is preceded by both a preprocessing and a normalisation operation of the RL observationsot. Firstly, because the high-frequency noise present in the trading data was experimen-tally observed to lower the algorithm generalisation, a low-pass filtering operation is executed. However, such a preprocessing operation has a cost as it mod-ifies or even destroys some potentially useful trading patterns and introduces a non-negligible lag. Sec-ondly, the resulting data are transformed in order to convey more meaningful information about market movements. Typically, the daily evolution of prices is considered rather than the raw prices. Thirdly, the remaining data are normalised. •Data augmentation techniques: A key challenge of this algorithmic trading problem is the limited amount of available data, which are in addition gen-erally of poor quality. As a counter to this ma-jor problem, several data augmentation techniques are implemented: signal shifting, signal filtering and artificial noise addition. The application of such data augmentation techniques will artificially gen-erate new trading data which are slightly different but which result in the same financial phenomena. Finally, the algorithm underneath the TDQN trading strategy is depicted in detail in Algorithm 1.

5. Performance assessment

### An accurate performance evaluation approach is capi-

### tion, a novel, more reliable methodology is presented to

9

---

```text
Algorithm 1 TDQN algorithm
Initialise the experience replay memory Mof capacity C.
Initialise the main DNN weightsθ (Xavier initialisation).
Initialise the target DNN weightsθ=θ.
```

### −

```text
for episode = 1 to Ndo
Acquire the initial observationo 1 from the environment Eand preprocess it.
fort = 1 to Tdo
With probability, select a random actionatfrom A.
Otherwise, selectat= arg maxa∈AQ (ot, a; θ).
Copy the environment E =E.
```

### −

```text
Interact with the environment E (actionat) and get the new observationot+1 and rewardrt.
Perform the same operation on E with the opposite actionat, gettingot+1 andrt.
```

### − − − −

```text
Preprocess both new observationsot+1 andot+1.
```

### −

```text
Store both experienceset= (ot, at, rt, ot+1) andet= (ot, at, rt, ot+1) in M.
```

### − − − −

```text
ift % T’ = 0 then
Randomly sample from Ma minibatch of Neexperiencesei= (oi, ai, ri, oi+1).
Setyi=
{
```

### ri if the statesi+1 is terminal,

```text
ri+γ Q (oi+1, arg maxa∈AQ (oi+1, a; θ); θ) otherwise.
```

### −

```text
Compute and clip the gradients based on the Huber loss H (yi, Q (oi, ai; θ)).
Optimise the main DNN parametersθbased on these clipped gradients.
Update the target DNN parametersθ=θevery N steps.
```

### − −

> end if
> Anneal the \-Greedy exploration parameter.
> end for
> end for

objectively assess the performance of algorithmic trading rejected, even though it could be interesting to assess the strategies, including the TDQN algorithm. robustness of trading strategies with respect to such an 5.1. Testbench

In the literature, the performance of a trading strategy contain significant market regime shifts which would seri-is generally assessed on a single instrument (stock marously harm the training stability of the trading strategies. ket or others) for a certain period of time. Nevertheless, the analysis resulting from such a basic approach should both training and test sets as follows: not be entirely trusted, as the trading data could have been specifically selected so that a trading strategy looks profitable, even though it is not the case in general. To •Test set: 01/01/2018→31/12/2019. eliminate such bias, the performance should ideally be as-sessed on multiple instruments presenting diverse patterns. A validation set is also considered as a subset of the Aiming to produce trustful conclusions, this research patraining set for the tuning of the numerous TDQN algo-per proposes a testbench composed of 30 stocks presenting rithm hyperparameters. Note that the RL policy DNN diverse characteristics (sectors, regions, volatility, liquidparametersθare fixed during the execution of the trading ity, etc.). The testbench is depicted in Table 2. To avoid strategy on the entire test set, meaning that the new ex-any confusion, the official reference for each stock (ticker) periences acquired are not valued for extra training. Nev-is specified in parentheses. To avoid any ambiguities conertheless, such practice constitutes an interesting future cerning the training and evaluation protocols, it should be research direction. mentioned that a new trading strategy is trained for each stock included in the testbench. Nevertheless, for the sake To end this subsection, it should be noted that the pro-of generality, all the algorithm hyperparameters remain posed testbench could be improved thanks to even more unchanged over the entire testbench. diversification. The obvious addition would be to include Regarding the trading horizon, the eight years precederties. Another interesting addition would be to consider ing the publication year of the research paper are selected different training/testing time periods while excluding the to be representative of the current market conditions. Such significant market regime shifts. Nevertheless, this last a short-time period could be criticised because it may be idea was discarded in this scientific article due to the im-too limited to be representative of the entire set of finanportant time already required to produce results for the cial phenomena. For instance, the financial crisis of 2008 is proposed testbench.

extraordinary event. However, this choice was motivated by the fact that a shorter trading horizon is less likely to Finally, the trading horizon of eight years is divided into

```text
•Training set: 01/01/2012→31/12/2017.
```

### more stocks with different financial situations and prop-

10

---

### Table 2: Performance assessment testbench

### Sector Region

### American European Asian

### Trading index S&P 500 (SPY)

![Page 11 image](images/page-011-image-01.png)

> Dow Jones (DIA) FTSE 100 (EZU) Nikkei 225 (EWJ)
> NASDAQ (QQQ)

Technology Amazon (AMZN) Siemens (SIE.DE) Tencent (0700.HK)

> Apple (AAPL) Nokia (NOK) Sony (6758.T)
> Google (GOOGL) Philips (PHIA.AS) Baidu (BIDU)
> Facebook (FB) Alibaba (BABA)
> Microsoft (MSFT)
> Twitter (TWTR)

Financial services JPMorgan Chase (JPM) HSBC (HSBC) CCB (0939.HK)

### Energy Exxon Mobil (XOM) Shell (RDSA.AS) Petro China (PTR)

![Page 11 image](images/page-011-image-02.png)

Automotive Tesla (TSLA) Volkswagen (VOW3.DE) Toyota (7203.T) Food Coca Cola (KO) AB In Bev (ABI.BR) Kirin (2503.T)

5.2. Benchmark trading strategies mean reversion strategy does not, the opposite being true In order to properly assess the strengths and weakas well. This is due to the fact that these two families nesses of the TDQN algorithm, some benchmark algoof trading strategies adopt opposite positions: a mean re-rithmic trading strategies were selected for comparison version strategy always denies and goes against the trends purposes. Only the classical trading strategies commonly while a trend following strategy follows the movements. used in practice were considered, excluding for instance strategies based on DL techniques or other advanced ap-proaches. Despite the fact that the TDQN algorithm is an active trading strategy, both passive and active strate-gies are taken into consideration. For the sake of fairness, the strategies share the same input and output spaces pre-sented in Section 3.4.2 (Oand A). The following list sum-marises the benchmark strategies selected: •Buy and hold (B&H). •Sell and hold (S&H). •Trend following with moving averages (TF). •Mean reversion with moving averages (MR). For the sake of brevity, a detailed description of each strategy is not provided in this research paper. The reader can refer to Chan (2009), Chan (2013) or Narang (2009) for more information. The first two benchmark trading strategies (B&H and S&H) are said to be passive, as there are no changes in trading position over the trading hori-zon. On the contrary, the other two benchmark strategies 5.3. Quantitative performance assessment (TF and MR) are active trading strategies, issuing multi The quantitative performance assessment consists in ple changes in trading positions over the trading horizon. defining one performance indicator or more to numerically On the one hand, a trend following strategy is concerned quantify the performance of an algorithmic trading strat-with the identification and the follow-up of significant maregy. Because the core objective of a trading strategy is ket trends, as depicted in Figure 5. On the other hand, a to be profitable, its performance should be linked to the mean reversion strategy, illustrated in Figure 6, is based amount of money earned. However, such reasoning omits on the tendency of a stock market to get back to its previto consider the risk associated with the trading activity ous average price in the absence of clear trends. By design, which should be efficiently mitigated. Generally, a trading a trend following strategy generally makes a profit when a strategy achieving a small but stable profit is preferred to

> Figure 5: Illustration of a typical trend following trading strategy
> Figure 6: Illustration of a typical mean reversion trading strategy

11

---

### Table 3: Quantitative performance assessment indicators

### Performance indicator Description

Sharpe ratio Return of the trading activity compared to its riskiness. Profit & loss Money gained or lost at the end of the trading activity. Annualised return Annualised return generated during the trading activity. Annualised volatility Modelling of the risk associated with the trading activity. Profitability ratio Percentage of winning trades made during the trading activity. Profit and loss ratio Ratio between the trading activity trades average profit and loss. Sortino ratio Similar to the Sharpe ratio with the negative risk penalised only. Maximum drawdown Largest loss from a peak to a trough during the trading activity. Maximum drawdown duration Time duration of the trading activity maximum drawdown.

a trading strategy achieving a huge profit in a very unstahttps://github.com/Thibaut Theate/An-Applicat
ble way after suffering from multiple losses. It eventually ion-of-Deep-Reinforcement-Learning-to-Algorith

depends on the investor profile and the willingness to take mic-Trading. extra risks to potentially earn more.

Multiple performance indicators were selected to accu The first detailed analysis concerns the execution of the rately assess the performance of a trading strategy. As TDQN trading strategy on the Apple stock, resulting in previously introduced in Section 3.5, the most important promising results. Similar to many DRL algorithms, the one is certainly the Sharpe ratio. This performance in TDQN algorithm is subject to a non-negligible variance. dicator, widely used in the field of algorithmic trading, Multiple training experiments with the exact same initial is particularly informative as it combines both profitabilconditions will inevitably lead to slightly different trading ity and risk. Besides the Sharpe ratio, this research paper strategies of varying performance. As a consequence, both considers multiple other performance indicators to provide a typical run of the TDQN algorithm and its expected per-

extra insights. Table 3 presents the entire set of perforformance are presented hereafter. mance indicators employed to quantify the performance of a trading strategy.

Complementarily to the computation of these numerinitial amount of money being equal to$100, 000. The ous performance indicators, it is interesting to graphically TDQN algorithm achieves good results from both an earn-represent the trading strategy behaviour. Plotting both ings and a risk mitigation point of view, clearly outper-the stock market priceptand portfolio valuevtevolutions forming all the benchmark active and passive trading strate-together with the trading actionsatissued by the trading gies. Secondly, Figure 7 plots both the stock market price strategy seems appropriate to accurately analyse the tradptand RL agent portfolio valuevtevolutions, together ing policy. Moreover, such visualisation could also provide with the actionsatoutputted by the TDQN algorithm. It extra insights about the performance, the strengths and can be observed that the DRL trading strategy is capable

weaknesses of the strategy analysed.

6. Results and discussion

In this section, the TDQN trading strategy is evaluket trends, meaning that the TDQN algorithm learned to ated following the performance assessment methodology be more reactive than proactive for this particular stock. previously described. Firstly, a detailed analysis is per This behaviour is expected with such a limited observation formed for both a case that give good results and a case space Onot including the reasons for the future market for which the results were mitigated. This highlights the directions (new product announcement, financial report, strengths, weaknesses and limitations of the TDQN algomacroeconomics, etc.). However, this does not mean that rithm. Secondly, the performance achieved by the DRL the policies learned are purely reactive. Indeed, it was ob-trading strategy on the entire testbench is summarised served that the RL agent may decide to adapt its trading and analysed. Finally, some additional discussions about position before a trend inversion by noticing an increase the discount factor parameter, the trading costs influence in volatility, therefore anticipating and being proactive. are provided. The experimental code supporting the re Expected performance: In order to estimate the ex-sults presented is publicly available at the following link: pected performance as well as the variance of the TDQN

### and the main challenges faced by the TDQN algorithm

### 6.1. Good results \- Apple stock

Typical run: Firstly, Table 4 presents the perfor-mance achieved by each trading strategy considered, the of accurately detecting and benefiting from major trends, while being more hesitant during market behavioural shifts when the volatility increases. It can also be seen that the trading agent generally lags slightly behind the mar-algorithm, the same RL trading agent is trained multiple

12

---

### Table 4: Performance assessment for the Apple stock

Performance indicator B&H S&H TF MR TDQN

Sharpe ratio 1.239 \-1.593 1.178 \-0.609 1.484 Profit & loss [$] 79823 \-80023 68738 \-34630 100288 Annualised return [%] 28.86 \-100.00 25.97 \-19.09 32.81 Annualised volatility [%] 26.62 44.39 24.86 28.33 25.69 Profitability ratio [%] 100 0.00 42.31 56.67 52.17 Profit and loss ratio ∞ 0.00 3.182 0.492 2.958 Sortino ratio 1.558 \-2.203 1.802 \-0.812 1.841 Max drawdown [%] 38.51 82.48 14.89 51.12 17.31 Max drawdown duration [days] 62 250 20 204 25

### Figure 7: TDQN algorithm execution for the Apple stock (test set)

![Page 13 image](images/page-013-image-01.png)

times. Figure 8 plots the averaged (over 50 iterations) per-formance of the TDQN algorithm for both the training and test sets with respect to the number of training episodes. This expected performance is comparable to the perfor-mance achieved during the typical run of the algorithm. It can also be noticed that the overfitting tendency of the RL agent seems to be properly handled for this specific market. Please note that the test set performance being temporarily superior to the training set performance is not a mistake. It simply indicates an easier to trade and more profitable market for the test set trading period for the Apple stock. This example perfectly illustrates a major difficulty of the algorithmic trading problem: the train-ing and test sets do not share the same distributions. In-deed, the distribution of the daily returns is continuously changing, which complicates both the training of the DRL trading strategy and its performance evaluation.

### 6.2. Mitigated results \- Tesla stock

```text
The same detailed analysis is performed on the Tesla quency (changes in trading positions, which correspond to
stock, which presents very different characteristics comthe situation whereat 6=at−1) despite the non-negligible
pared to the Apple stock, such as a pronounced volatility. trading costs, which increases even more the riskiness of
```

![Page 13 image](images/page-013-image-02.png)

In contrast to the promising performance achieved on the the DRL trading strategy. previous stock, this case was specifically selected to high-light the limitations of the TDQN algorithm.

### Figure 8: TDQN algorithm expected performance for the Apple stock

Typical run: Similar to the previous analysis, Ta-ble 5 presents the performance achieved by every trad-ing strategies considered, the initial amount of money be-ing equal to$100, 000. The mitigated results achieved by the benchmark active strategies suggest that the Tesla stock is quite difficult to trade, which is partly due to its significant volatility. Even though the TDQN algorithm achieves a positive Sharpe ratio, almost no profit is gen-erated. Moreover, the risk level associated with this trad-ing activity cannot really be considered acceptable. For instance, the maximum drawdown duration is particularly large, which would result in a stressful situation for the op-erator responsible for the trading strategy. Figure 9, which plots both the stock market priceptand RL agent port-folio valuevtevolutions together with the actionsatout-putted by the TDQN algorithm, confirms this observation. Moreover, it can be clearly observed that the pronounced volatility of the Tesla stock induces a higher trading fre-

13

---

### Table 5: Performance assessment for the Tesla stock

Performance indicator B&H S&H TF MR TDQN

Sharpe ratio 0.508 \-0.154 \-0.987 0.358 0.261 Profit & loss [$] 29847 \-29847 \-73301 8600 98 Annualised return [%] 24.11 \-7.38 \-100.00 19.02 12.80 Annualised volatility [%] 53.14 46.11 52.70 58.05 52.09 Profitability ratio [%] 100 0.00 34.38 67.65 38.18 Profit and loss ratio ∞ 0.00 0.534 0.496 1.621 Sortino ratio 0.741 \-0.205 \-1.229 0.539 0.359 Max drawdown [%] 52.83 54.09 79.91 65.31 58.95 Max drawdown duration [days] 205 144 229 159 331

Figure 9: TDQN algorithm execution for the Tesla stock (test set) Figure 10: TDQN algorithm expected performance for the Tesla

### Expected performance: Figure 10 plots the expected

![Page 14 image](images/page-014-image-01.png)

performance of the TDQN algorithm for both the trainin Section 5.1, in order to draw more robust and trust-ing and test sets as a function of the number of training ful conclusions. Table 6 presents the expected Sharpe ra-that this expected performance is significantly better than strategies on the entire set of stocks included in this test-a key limitation of the TDQN algorithm: the substantial trading strategies, it is important to differentiate the pas-variance which may result in selecting poorly performing sive strategies (B&H and S&H) from the active ones (TF policies compared to the expected performance. The sigand MR). Indeed, this second family of trading strategies also suggests that the DRL algorithm is subject to overfitrisk: continuous speculation. Because the stock markets techniques implemented. This overfitting phenomenon can with some instabilities during the test set trading period, too limited to efficiently apprehend the Tesla stock. Even forming the other benchmark trading strategies. In fact, though this overfitting phenomenon does not seem to be neither the trend following nor the mean reversion strat-too harmful in this particular case, it may lead to poor egy managed to generate satisfying results on average on

episodes (over 50 iterations). It can be directly noticed the performance achieved by the typical run previously bench. analysed, which can therefore be considered as not really representative of the average behaviour. This highlights

nificantly higher performance achieved on the training set has more potential at the cost of an extra non-negligible ting in this specific case, despite the multiple regularisation were mostly bullish (priceptmainly increasing over time) be partially explained by the observation space Owhich is

performance for other stocks. 6.3. Global results \- Testbench As previously suggested in this research paper, the TDQN algorithm is evaluated on the testbench introduced

![Page 14 image](images/page-014-image-02.png)

### stock

### tio achieved by both the TDQN and benchmark trading

### Regarding the performance achieved by the benchmark

it is not surprising to see the buy and hold strategy outper-this testbench. It clearly indicates that there is a major difficulty to actively trade in such market conditions. This poorer performance can also be explained by the fact that such strategies are generally well suited to exploit specific financial patterns, but they lack versatility and thus of-ten fail to achieve good average performance on a large

14

---

> Table 6: Performance assessment for the entire testbench

```text
Stock Sharpe Ratio
```

B&H S&H TF MR TDQN

Dow Jones (DIA) 0.684 \-0.636 \-0.325 \-0.214 0.684

S&P 500 (SPY) 0.834 \-0.833 \-0.309 \-0.376 0.834 NASDAQ 100 (QQQ) 0.845 \-0.806 0.264 0.060 0.845 FTSE 100 (EZU) 0.088 0.026 \-0.404 \-0.030 0.103 Nikkei 225 (EWJ) 0.128 \-0.025 \-1.649 0.418 0.019 Google (GOOGL) 0.570 \-0.370 0.125 0.555 0.227 Apple (AAPL) 1.239 \-1.593 1.178 \-0.609 1.424 Facebook (FB) 0.371 \-0.078 0.248 \-0.168 0.151 Amazon (AMZN) 0.559 \-0.187 0.161 \-1.193 0.419 Microsoft (MSFT) 1.364 \-1.390 \-0.041 \-0.416 0.987 Twitter (TWTR) 0.189 0.314 \-0.271 \-0.422 0.238 Nokia (NOK) \-0.408 0.565 1.088 1.314 \-0.094 Philips (PHIA.AS) 1.062 \-0.672 \-0.167 \-0.599 0.675 Siemens (SIE.DE) 0.399 \-0.265 0.525 0.526 0.426 Baidu (BIDU) \-0.699 0.866 \-1.209 0.167 0.080 Alibaba (BABA) 0.357 \-0.139 \-0.068 0.293 0.021 Tencent (0700.HK) \-0.013 0.309 0.179 \-0.466 \-0.198 Sony (6758.T) 0.794 \-0.655 \-0.352 0.415 0.424 JPMorgan Chase (JPM) 0.713 \-0.743 \-1.325 \-0.004 0.722 HSBC (HSBC) \-0.518 0.725 \-1.061 0.447 0.011 CCB (0939.HK) 0.026 0.165 \-1.163 \-0.388 0.202 Exxon Mobil (XOM) 0.055 0.132 \-0.386 \-0.673 0.098 Shell (RDSA.AS) 0.488 \-0.238 \-0.043 0.742 0.425 Petro China (PTR) \-0.376 0.514 \-0.821 \-0.238 0.156 Tesla (TSLA) 0.508 \-0.154 \-0.987 0.358 0.621 Volkswagen (VOW3.DE) 0.384 \-0.208 \-0.361 0.601 0.216 Toyota (7203.T) 0.352 \-0.242 \-1.108 \-0.378 0.304 Coca Cola (KO) 1.031 \-0.871 \-0.236 \-0.394 1.068 AB In Bev (ABI.BR) \-0.058 0.275 0.036 \-1.313 0.187 Kirin (2503.T) 0.106 0.156 \-1.441 0.313 0.852

```text
Average 0.369 -0.202 -0.331 -0.056 0.404
```

15

---

set of stocks presenting diverse characteristics. Moreover, 6.5. Trading costs discussion research paper).

such strategies are generally more impacted by the trad The analysis of the trading costs influence on a trading ing costs due their higher trading frequency (for relatively strategy behaviour and performance is capital, due to the short moving averages durations, as it is the case in this fact that such costs represent an extra risk to mitigate. A

Concerning the innovative trading strategy, the TDQN DL architectures is related to the trading costs. As pre-algorithm achieves promising results on the testbench, outviously explained in Section 3, the RL formalism enables performing the benchmark active trading strategies on avthe consideration of these additional costs directly into the erage. Nevertheless, the DRL trading strategy only barely decision-making process. The optimal policy is learned surpasses the buy and hold strategy on these particular according to the trading costs value. On the contrary, bullish markets which are so favourable to this simple pasa purely predictive approach would only output predic-sive strategy. Interestingly, it should be noted that the tions about the future market direction or prices without performance of the TDQN algorithm is identical or very any indications regarding an appropriate trading strategy close to the performance of the passive trading strategies taking into account the trading costs. Although this last (B&H and S&H) for multiple stocks. This is explained by approach offers more flexibility and could certainly lead the fact that the DRL strategy efficiently learns to tend to well-performing trading strategies, it is less efficient by emphasized that the TDQN algorithm is neither a trend In order to illustrate the ability of the TDQN algo-following nor a mean reversion trading strategy as both rithm to automatically and efficiently adapt to different financial patterns can be efficiently handled in practice. trading costs, Figure 11 presents the behaviour of the DRL Thus, the main advantage of the DRL trading strategy is trading strategy for three different costs values, all other certainly its versatility and its ability to efficiently handle parameters remaining unchanged. It can clearly be ob-various markets presenting diverse characteristics. served that the TDQN algorithm effectively reduces its

toward a passive trading strategy when the uncertainty design. associated to active trading increases. It should also be

### 6.4. Discount factor discussion

As previously explained in Section 3.4, the discount rithm simply stops actively trading and adopts a passive factorγis concerned with the importance of future reapproach (buy and hold or sell and hold strategies). the significant uncertainty of the future. On the one hand, Nowadays, the main DRL solutions successfully ap-the desired trading policy should be long-term oriented plied to real-life problems concern specific environments (γ→1), in order to avoid a too high trading frequency with particular properties such as games (see e.g. the fa-and being exposed to considerable trading costs. On the mous Alpha Go algorithm developed by Google Deepmind other hand, it would be unwise to place too much im Silver et al. (2016)). In this research paper, an entirely portance on a stock market future which is particularly different environment characterised by a significant com-uncertain (γ→0). Therefore, a trade-off intuitively exists plexity and a considerable uncertainty is studied with the for the discount factor parameter. algorithmic trading problem. Obviously, multiple chal-This reasoning is validated by the multiple experiments algorithm, the major ones being summarised hereafter. served that there is an optimal value for the discount Firstly, the extremely poor observability of the trad-factor, which is neither too small nor too large. Addiing environment is a characteristic that significantly lim-tionally, these experiments highlighted the hidden link beits the performance of the TDQN algorithm. Indeed, the tween the discount factor and the trading frequency, due amount of information at the disposal of the RL agent to the trading costs. From the point of view of the RL is really not sufficient to accurately explain the financial agent, these costs represent an obstacle to overcome for a phenomena occurring during training, which is necessary change in trading position to occur, due to the immediate to efficiently learn to trade. Secondly, although the distri-reduced (and often negative) reward received. It models bution of the daily returns is continuously changing, the the fact that the trading agent should be sufficiently conpast is required to be representative enough of the future fident about the future in order to overcome the extra risk for the TDQN algorithm to achieve good results. This associated with the trading costs. The discount factor demakes the DRL trading strategy particularly sensitive to termining the importance assigned to the future, a small significant market regime shifts. Thirdly, the TDQN al-value for the parameterγwill inevitably reduce the tengorithm overfitting tendency has to be properly handled dency of the RL agent to change its trading position, which in order to obtain a reliable trading strategy. As sug-decreases the trading frequency of the TDQN algorithm. gested in Zhang et al. (2018), more rigorous evaluation

```text
wards. In the scope of this algorithmic trading problem,
the proper tuning of this parameter is not trivial due to 6.6. Core challenges
performed to tune the parameterγ. Indeed, it was ob-
```

major motivation for studying DRL solutions rather than pure prediction techniques that could also be based on trading frequency when the trading costs increase, as ex-pected. When these costs become too high, the DRL algo-lenges were faced during the research around the TDQN

16

---

> (a) Trading costs: 0% (b) Trading costs: 0.1% (c) Trading costs: 0.2%

> Figure 11: Impact of the trading costs on the TDQN algorithm, for the Apple stock

protocols are required in RL due to the strong tendency (2017). Another interesting research direction is the com-of common DRL techniques to overfit. More research on parison of the TDQN algorithm with Policy Optimisation this particular topic is required for DRL techniques to fit DRL algorithms such as the Proximal Policy Optimisation it rather difficult to successfully apply these algorithms to The last major research direction suggested concerns certain problems, especially when the training and test sets the formalisation of the algorithmic trading problem into a differ considerably. This is a key limitation of the TDQN reinforcement learning one. Firstly, the observation space algorithm which was previously highlighted for the Tesla Oshould be extended to enhance the observability of the

a broader range of real-life applications. Lastly, the sub (PPO \- Schulman et al. (2017)) algorithm.
stantial variance of DRL algorithms such as DQN makes

stock.

![Page 17 image](images/page-017-image-01.png)

7. Conclusion

This scientific research paper presents the Trading Deep tween the RL objective and the Sharpe ratio maximisation Q-Network algorithm (TDQN), a deep reinforcement learnobjective. Finally, an interesting and promising research ing (DRL) solution to the algorithmic trading problem direction is the consideration of distributions instead of ex-of determining the optimal trading position at any point pected values in the TDQN algorithm in order to encom-the TDQN algorithm demonstrates multiple benefits com Thibaut Th ́eate is a Research Fellow of the F.R.S.-

in time during a trading activity in stock markets. Fol-lowing a rigorous performance assessment, this innova-tive trading strategy achieves promising results, surpassing Acknowledgements on average the benchmark trading strategies. Moreover, pared to more classical approaches, such as an appreciable FNRS, of which he acknowledges the financial support. versatility and a remarkable robustness to diverse trading costs. Additionally, such data-driven approach presents the major advantage of suppressing the complex task of markets considered.

defining explicit rules suited to the particular financial Ar ́evalo, A., Ni ̃no, J., Hern ́andez, G., and Sandoval, J. (2016). High-

Nevertheless, the performance of the TDQN algorithm A. A. (2017). A Brief Survey of Deep Reinforcement Learning.

![Page 17 image](images/page-017-image-02.png)

could still be improved, from both a generalisation and a Co RR, abs/1708.05866. reproducibility point of view, to cite a few. Several re-search directions are suggested to upgrade the DRL solu-tion, such as the use of LSTM layers into the deep neural the American Mathematical Society, pages 458-471. time-series data, see e.g. Hausknecht and Stone (2015). Another example is the consideration of the numerous im-are detailed in Sutton and Barto (2018), van Hasselt et al. abs/1707.06887. et al. (2017), Fortunato et al. (2018) and Hessel et al.

network which should help to better process the financial Bao, W. N., Yue, J., and Rao, Y. (2017). A Deep Learning Frame-
provements implemented in the Rainbow algorithm, which tributional Perspective on Reinforcement Learning. Co RR,
(2015), Wang et al. (2015), Schaul et al. (2016), Bellemare Bollen, J., Mao, H., and jun Zeng, X. (2011). Twitter Mood Predicts

trading environment. Similarly, some constraints about the action space Acould be relaxed in order to enable new trading possibilities. Secondly, advanced RL reward engineering should be performed to narrow the gap be-pass the notion of risk and to better handle uncertainty.

### References

![Page 17 image](images/page-017-image-03.png)

> Frequency Trading Strategy Based on Deep Neural Networks.

> ICIC.

> Arulkumaran, K., Deisenroth, M. P., Brundage, M., and Bharath,
> Bailey, D. H., Borwein, J. M., de Prado, M. L., and Zhu, Q. J. (2014).
> Pseudo-Mathematics and Financial Charlatanism: The Effects of
> Backtest Overfitting on Out-of-Sample Performance. Notice of
> work for Financial Time Series using Stacked Autoencoders and
> Long-Short Term Memory. Plo S one, 12.
> Bellemare, M. G., Dabney, W., and Munos, R. (2017). A Dis-

> the Stock Market. J. Comput. Science, 2: 1-8.

17

---

Boukas, I., Ernst, D., Th ́eate, T., Bolland, A., Huynen, A., Buchtized Experience Replay. Co RR, abs/1511.05952.
wald, M., Wynants, C., and Corn ́elusse, B. (2020). A Deep Rein Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov,
forcement Learning Framework for Continuous Intraday Market O. (2017). Proximal Policy Optimization Algorithms. Co RR,
Bidding. Ar Xiv, abs/2004.05940.

Busoniu, L., Babuska, R., De Schutter, B., and Ernst, D. (2010). Re Shao, K., Tang, Z., Zhu, Y., Li, N., and Zhao, D. (2019). A Sur-inforcement Learning and Dynamic Programming using Function vey of Deep Reinforcement Learning in Video Games. Ar Xiv, Approximators. CRC Press.

Carapu ̧co, J., Neves, R. F., and Horta, N. (2018). Reinforcement Silver, D., Huang, A., Maddison, C. J., Guez, A., Sifre, L., van den
Learning applied to Forex Trading. Appl. Soft Comput., 73: 783 Driessche, G., Schrittwieser, J., Antonoglou, I., Panneershelvam,
794.

Chan, E. P. (2009). Quantitative Trading: How to Build Your Own ner, N., Sutskever, I., Lillicrap, T. P., Leach, M., Kavukcuoglu,
Algorithmic Trading Business. Wiley. K., Graepel, T., and Hassabis, D. (2016). Mastering the Game of
Chan, E. P. (2013). Algorithmic Trading: Winning Strategies and Go with Deep Neural Networks and Tree Search. Nature, 529: 484-
Their Rationale. Wiley.

Dempster, M. A. H. and Leemans, V. (2006). An Automated FX Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Trading System using Adaptive Reinforcement Learning. Expert Introduction. The MIT Press, second edition. Syst. Appl., 30: 543-552. Deng, Y., Bao, F., Kong, Y., Ren, Z., and Dai, Q. (2017). Deep gan and Claypool Publishers. Direct Reinforcement Learning for Financial Signal Representa Treleaven, P. C., Galas, M., and Lalchand, V. (2013). Algorithmic tion and Trading. IEEE Transactions on Neural Networks and Trading Review. Commun. ACM, 56: 76-85. Learning Systems, 28: 653-664.

Fortunato, M., Azar, M. G., Piot, B., Menick, J., Hessel, M., Osband, ment Learning with Double Q-Learning. Co RR, abs/1509.06461.
I., Graves, A., Mnih, V., Munos, R., Hassabis, D., Pietquin, O., Wang, Z., de Freitas, N., and Lanctot, M. (2015). Dueling Net-
Blundell, C., and Legg, S. (2018). Noisy Networks for Exploration. work Architectures for Deep Reinforcement Learning. Co RR,
Co RR, abs/1706.10295.

Goodfellow, I., Bengio, Y., and Courville, A. (2016). Deep Learning. Watkins, C. J. C. H. and Dayan, P. (1992). Technical Note: Q-
MIT Press.

Goodfellow, I. J., Bengio, Y., and Courville, A. C. (2015). Deep Zhang, C., Vinyals, O., Munos, R., and Bengio, S. (2018). A
Learning. Nature, 521: 436-444.
Hausknecht, M. J. and Stone, P. (2015). Deep Recurrent Q-Learning abs/1804.06893.
for Partially Observable MDPs. Co RR, abs/1507.06527.
Hendershott, T., Jones, C. M., and Menkveld, A. J. (2011). Does
Algorithmic Trading Improve Liquidity? Journal of Finance,
66: 1-33.
Hessel, M., Modayil, J., van Hasselt, H. P., Schaul, T., Ostrovski,
G., Dabney, W., Horgan, D., Piot, B., Azar, M. G., and Silver, D.
(2017). Rainbow: Combining Improvements in Deep Reinforce-
ment Learning. Co RR, abs/1710.02298.
Ioannidis, J. P. A. (2005). Why Most Published Research Findings
Are False. PLo S Med, 2: 124.
Ioffe, S. and Szegedy, C. (2015). Batch Normalization: Accelerat-
ing Deep Network Training by Reducing Internal Covariate Shift.
Co RR, abs/1502.03167.
Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic
Optimization. Co RR, abs/1412.6980.
Le Cun, Y., Bengio, Y., and Hinton, G. (2015). Deep Learning. Na-
ture, 521.
Leinweber, D. and Sisk, J. (2011). Event-Driven Trading and the
“New News”. The Journal of Portfolio Management, 38: 110-124.
Li, Y. (2017). Deep Reinforcement Learning: An Overview. Co RR,
abs/1701.07274.
Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I.,
Wierstra, D., and Riedmiller, M. A. (2013). Playing Atari with
Deep Reinforcement Learning. Co RR, abs/1312.5602.
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J.,
Bellemare, M. G., Graves, A., Riedmiller, M. A., Fidjeland, A.,
Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou,
I., King, H., Kumaran, D., Wierstra, D., Legg, S., and Hassabis,
D. (2015). Human-Level Control through Deep Reinforcement
Learning. Nature, 518: 529-533.
Moody, J. E. and Saffell, M. (2001). Learning to Trade via Direct
Reinforcement. IEEE transactions on neural networks, 12 4: 875-
89.
Narang, R. K. (2009). Inside the Black Box. Wiley.
Nuij, W., Milea, V., Hogenboom, F., Frasincar, F., and Kaymak, U.
(2014). An Automated Framework for Incorporating News into
Stock Trading Strategies. IEEE Transactions on Knowledge and
Data Engineering, 26: 823-835.
Nuti, G., Mirghaemi, M., Treleaven, P. C., and Yingsaeree, C.
(2011). Algorithmic Trading. Computer, 44: 61-69.
Schaul, T., Quan, J., Antonoglou, I., and Silver, D. (2016). Priori-

> abs/1707.06347.

> abs/1912.10944.

> V., Lanctot, M., Dieleman, S., Grewe, D., Nham, J., Kalchbren-

> 489.

> Szepesvari, C. (2010). Algorithms for Reinforcement Learning. Mor-
> van Hasselt, H. P., Guez, A., and Silver, D. (2015). Deep Reinforce-

> abs/1511.06581.

> Learning. Machine Learning, 8: 279-292.

### Study on Overfitting in Deep Reinforcement Learning. Co RR,

18

---

```text
Appendix A. Derivation of action space A Case of Qt≥0: The previous expression becomes
Theorem 1. The RL action space Aadmits an upper ⇔v≥ −ntpt (1 +C)(1 +)−Qtpt (1 +C)
```

### bound Qtsuch that:

### Qt= vc

*t*

### pt (1 +C)

```text
Proof. The upper bound of the RL action space Ais de Case of Qt<0: The previous expression becomes
rived from the fact that the cash valuevtchas to remain vtc−Qtpt+C Qtpt≥ −(nt+Qt) pt (1 +C)(1 +)
positive over the entire trading horizon (Equation 9). Mak⇔vt≥ −ntpt (1 +C)(1 +)−Qtpt (2C+ + C)
traded by the RL agent at time stepthas to be set such The expression on the right side of the inequality repre-
```

```text
ing the hypothesis thatvc≥0, the number of shares Qt ⇔Q≥
```

*t*

### thatvc ≥0 as well. Introducing this condition into

### t+1

Equation 12 expressing the update of the cash value, the following expression is obtained:

### vt−Qtpt−C|Qt|pt≥0

*c*

```text
Two cases arise depending on the value of Qt:
Case of Qt<0: The previous expression becomes
vtc−Qtpt+C Qtpt≥0.
⇔Qt≤.
```

> vt
> c

### pt (1−C)

```text
The expression on the right side of the inequality is al-
ways positive due to the hypothesis thatvc≥0. Because
```

*t*

```text
Qtis negative in this case, the condition is always satisfied.
Case of Qt≥0: The previous expression becomes
vtc−Qtpt−C Qtpt≥0.
⇔Qt≤.
```

> vt
> c

### pt (1+C)

This condition represents the upper bound (positive) of tion 13 was verified for time stept. In this case, the most

the RL action space A.

```text
Theorem 2. The RL action space Aadmits a lower bound pt (2C+(1 +C))
Qtsuch that:
```

### Qt= ∆

### {

> ∆t
> pt (1+C)

> if∆t≥0
> t if∆t<0

### pt (2C+(1+C))

```text
with∆t=−vt−ntpt (1 +)(1 +C).
```

*c*

```text
Proof. The lower bound of the RL action space Ais de-
rived from the fact that the cash valuevchas to be suffi-
```

*t*

cient to get back to a neutral position (nt= 0) over the en-tire trading horizon (Equation 13). Making the hypothesis that this condition is satisfied at time stept, the number of shares Qttraded by the RL agent should be such that this condition remains true at the next time stept+ 1. In-troducing this constraint into Equation 12, the following inequality is obtained:

### vt−Qtpt−C|Qt|pt≥ −(nt+Qt) pt (1 +C)(1 +)

*c*

### Two cases arise depending on the value of Qt:

### vtc−Qtpt−C Qtpt≥ −(nt+Qt) pt (1 +C)(1 +)

*c*

*t*

### ⇔Qt≥

### −vt−ntpt (1+C)(1+)

*c*

### pt (1+C)

The expression on the right side of the inequality repre-sents the first lower bound for the RL action space A.

*c*

> t pt (2C+(1+C))
> −vt−ntpt (1+C)(1+)

*c*

sents the second lower bound for the RL action space A.

Both lower bounds previously derived have the same numerator, which is denoted ∆tfrom now on. This quan-tity represents the difference between the maximum as-sumed cost to get back to a neutral position at the next time stept+ 1 and the current cash value of the agentvt.

*c*

The expression tests whether the agent can pay its debt in the worst assumed case or not at the next time step, if nothing is done at the current time step (Qt= 0). Two cases arise depending on the sign of the quantity ∆t: Case of ∆t<0: The trading agent has no problem paying its debt in the situation previously described. This is al-ways true when the agent owns a positive number of shares (nt≥0). This is also always true when the agent owns a negative number of shares (nt<0) and when the price decreases (pt< pt−1) due to the hypothesis that Equa-constraining lower bound of the two is the following:

### Qt= ∆t

```text
Case of ∆t≥0: The trading agent may have problem pay-
ing its debt in the situation previously described. Follow-
ing a similar reasoning than for the previous case, the most
constraining lower bound of the two is the following:
```

### Qt= ∆t

### pt (1 +C)

19