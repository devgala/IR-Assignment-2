package test_server;
import io.javalin.Javalin;
import test_server.User;
import com.fasterxml.jackson.databind.ObjectMapper; 
import com.fasterxml.jackson.databind.ObjectWriter; 
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;

public class App {
    public String getGreeting() {
        return "Server Started";
    }

    public static void main(String[] args) throws JsonProcessingException {
        System.out.println(new App().getGreeting());
        ObjectWriter ow = new ObjectMapper().writer().withDefaultPrettyPrinter();
        String json = ow.writeValueAsString(new User("User 1","1234"));
        System.out.println(json);
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode jsonNode = objectMapper.readTree(json);
        var app = Javalin.create()
                .start(7070);
        app.get("/", ctx -> ctx.html("<h1>Hello World</h1>"));
        app.get("/bye", ctx -> ctx.html("<h1>bye world</h1>"));
        app.get("/user", ctx -> ctx.json(json));
    }
}
