import java.lang.annotation.*;

/**
 * 测试注解提取功能的Java类
 */
@Deprecated
@SuppressWarnings("unused")
public class AnnotationTest {
    private int value;

    /**
     * 构造方法
     * @param value 初始值
     */
    @Override
    public AnnotationTest(int value) {
        this.value = value;
    }

    /**
     * 获取值
     * @return 当前值
     */
    @Override
    @SuppressWarnings("unchecked")
    public int getValue() {
        return value;
    }

    /**
     * 设置值
     * @param value 新值
     */
    @Deprecated
    @SuppressWarnings({"unchecked", "rawtypes"})
    public void setValue(int value) {
        this.value = value;
    }

    /**
     * 自定义注解测试
     */
    @MyAnnotation(name = "test", value = 100)
    public void testCustomAnnotation() {
        System.out.println("Custom annotation test");
    }

    /**
     * 静态方法测试
     */
    @SuppressWarnings("static-access")
    public static void staticMethod() {
        AnnotationTest test = new AnnotationTest(0);
        System.out.println(test.value);
    }
}

/**
 * 自定义注解
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
@interface MyAnnotation {
    String name();
    int value();
}
