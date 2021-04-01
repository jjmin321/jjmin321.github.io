abstract class AccountCommand implements Command{
    protected Account account; //자식 클래스에서 사용 가능
    protected int money; //자식 클래스에서 사용 가능

    public void setAccount (Account account) {
        this.account = account;
    }
}  

class DepositCommand extends AccountCommand {
    public DepositCommand(int money) {
        this.money = money;
    }
    public void execute() {
        account.setMoney(account.getMoney() + money);
        System.out.println(money+" 원 입금 성공! 잔액: "+account.getMoney());
    }
}

class WithDrawCommand extends AccountCommand {
    public WithDrawCommand(int money) {
        this.money = money;
    }
    public void execute() {
        if (account.getMoney() > money) {
            account.setMoney(account.getMoney() - money);
            System.out.println(money+" 원 출금 성공! 잔액: "+account.getMoney());
        } else {
            System.out.println(money+"원 출금 실패! 출금 가능 금액: "+account.getMoney());
        }
    }
}
