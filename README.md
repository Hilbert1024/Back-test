# Back-test
This is a Back-test program. You can simulate buying and selling stocks in history.
+ Input: A matrix with grades of stocks. Matrix rows represent time while colunms represent stock codes.
+ Output: A figure denoting the equity curve of a portfolio.

## Example
Here is an example. A strategy is buying one stock (WLOG, chose 000001.SZ, Pingan Bank as an example) in open price everyday and selling it in close price tomorrow. After deducing transation tax and stamp tax in A market, we will obtain the following equity curve, where the barplot 
denotes the excess return of ZZ500.

![Test](https://github.com/Hilbert1984/Back-test/blob/master/figure/000001.SZ.jpg)

You can also trade a portfolio. See what would happen if you traded 50 stocks in random everday, from 2010 to 2018.

![Test](https://github.com/Hilbert1984/Back-test/blob/master/figure/random.jpg)

We also provide user interface for convenient use.

![Test](https://github.com/Hilbert1984/Back-test/blob/master/figure/UI.png)
