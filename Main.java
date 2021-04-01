public class Main {
    public static void main(String[] args) {
        Account account = new Account();
        account.deposit(2000);
        account.withdraw(1000);
        account.deposit(3000);
        account.withdraw(5000);
    }
}