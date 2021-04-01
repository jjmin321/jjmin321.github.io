import jdk.nashorn.internal.objects.annotations.Getter;

public class Account {
    private int money = 0;
    public int getMoney() {
        return this.money;
    }
    public void setMoney(int money) {
        this.money = money;
    }
    public void deposit(int money) {
        AccountCommand depositCommand = new DepositCommand(money);
        depositCommand.setAccount(this);
        depositCommand.execute();
    }
    public void withdraw(int money) {
        AccountCommand withdrawCommand = new WithDrawCommand(money);
        withdrawCommand.setAccount(this);
        withdrawCommand.execute();
    }
}
