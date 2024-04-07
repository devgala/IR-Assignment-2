package test_server;

public class User {
    private String name;
    private String number;
    public User(String name,String number){
        this.name = name;
        this.number = number;
    }
    public String getName(){
        return this.name;
    }
    public String getNumber(){
        return this.number;
    }
}
