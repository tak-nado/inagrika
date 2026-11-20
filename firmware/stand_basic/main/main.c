#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Пины
#define TRIG_LEFT          GPIO_NUM_7
#define ECHO_LEFT          GPIO_NUM_15
#define TRIG_RIGHT         GPIO_NUM_5
#define ECHO_RIGHT         GPIO_NUM_6

#define MOTOR_1_STEP       GPIO_NUM_42
#define MOTOR_1_DIR        GPIO_NUM_41
#define MOTOR_2_STEP       GPIO_NUM_0
#define MOTOR_2_DIR        GPIO_NUM_45
#define MOTORS_ENABLE      GPIO_NUM_40

#define I2C_SDA            GPIO_NUM_20
#define I2C_SCL            GPIO_NUM_19
#define MPU6050_ADDRESS    0x68

#define STOP_DISTANCE_CM   25.0f
#define STEP_FREQUENCY_HZ  400
#define STEP_DUTY          512

// Изменение уровеня, если один из двигателей вращается в обратную сторону.
#define MOTOR_1_FORWARD    1
#define MOTOR_2_FORWARD    0

#define STEP_TIMER         LEDC_TIMER_0
#define STEP_MODE          LEDC_LOW_SPEED_MODE
#define MOTOR_1_CHANNEL    LEDC_CHANNEL_0
#define MOTOR_2_CHANNEL    LEDC_CHANNEL_1

static char motion = 'S';
static bool safe_stop = false;
static bool mpu_found = false;

static float distance_left = 400.0f;
static float distance_right = 400.0f;

static i2c_master_bus_handle_t i2c_bus;
static i2c_master_dev_handle_t mpu6050;

static float read_distance(gpio_num_t trig, gpio_num_t echo)
{
    gpio_set_level(trig, 0);
    esp_rom_delay_us(2);
    gpio_set_level(trig, 1);
    esp_rom_delay_us(10);
    gpio_set_level(trig, 0);

    int64_t timeout = esp_timer_get_time() + 25000;

    while (gpio_get_level(echo) == 0) {
        if (esp_timer_get_time() >= timeout) {
            return 400.0f;
        }
    }

    int64_t pulse_start = esp_timer_get_time();
    timeout = pulse_start + 25000;

    while (gpio_get_level(echo) == 1) {
        if (esp_timer_get_time() >= timeout) {
            return 400.0f;
        }
    }

    int64_t pulse_time = esp_timer_get_time() - pulse_start;
    return (float)pulse_time * 0.0343f / 2.0f;
}

static void stop_motors(void)
{
    ledc_stop(STEP_MODE, MOTOR_1_CHANNEL, 0);
    ledc_stop(STEP_MODE, MOTOR_2_CHANNEL, 0);
    gpio_set_level(MOTORS_ENABLE, 1);
    motion = 'S';
}

static void start_step_signals(void)
{
    ledc_set_duty(STEP_MODE, MOTOR_1_CHANNEL, STEP_DUTY);
    ledc_update_duty(STEP_MODE, MOTOR_1_CHANNEL);

    ledc_set_duty(STEP_MODE, MOTOR_2_CHANNEL, STEP_DUTY);
    ledc_update_duty(STEP_MODE, MOTOR_2_CHANNEL);

    gpio_set_level(MOTORS_ENABLE, 0);
}

static void set_motion(char command)
{
    if (command == 'S') {
        safe_stop = false;
        stop_motors();
        printf("STATE,IDLE\n");
        return;
    }

    if (safe_stop) {
        printf("ERROR,send S before a new movement command\n");
        return;
    }

    bool obstacle = distance_left <= STOP_DISTANCE_CM ||
                    distance_right <= STOP_DISTANCE_CM;

    if (obstacle && command != 'B') {
        safe_stop = true;
        stop_motors();
        printf("STATE,SAFE_STOP,obstacle\n");
        return;
    }

    if (command == 'F') {
        gpio_set_level(MOTOR_1_DIR, MOTOR_1_FORWARD);
        gpio_set_level(MOTOR_2_DIR, MOTOR_2_FORWARD);
    } else if (command == 'B') {
        gpio_set_level(MOTOR_1_DIR, !MOTOR_1_FORWARD);
        gpio_set_level(MOTOR_2_DIR, !MOTOR_2_FORWARD);
    } else if (command == 'L') {
        gpio_set_level(MOTOR_1_DIR, !MOTOR_1_FORWARD);
        gpio_set_level(MOTOR_2_DIR, MOTOR_2_FORWARD);
    } else if (command == 'R') {
        gpio_set_level(MOTOR_1_DIR, MOTOR_1_FORWARD);
        gpio_set_level(MOTOR_2_DIR, !MOTOR_2_FORWARD);
    } else {
        return;
    }

    motion = command;
    esp_rom_delay_us(5);
    start_step_signals();
}

static void read_uart_command(void)
{
    uint8_t data[16];
    int length = uart_read_bytes(UART_NUM_0, data, sizeof(data), 0);

    for (int i = 0; i < length; i++) {
        char command = (char)toupper((int)data[i]);

        if (command == 'F' || command == 'B' || command == 'L' ||
            command == 'R' || command == 'S') {
            set_motion(command);
        }
    }
}

static bool start_mpu6050(void)
{
    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = I2C_SDA,
        .scl_io_num = I2C_SCL,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };

    if (i2c_new_master_bus(&bus_config, &i2c_bus) != ESP_OK) {
        return false;
    }

    i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU6050_ADDRESS,
        .scl_speed_hz = 100000,
    };

    if (i2c_master_bus_add_device(i2c_bus, &device_config, &mpu6050) != ESP_OK) {
        return false;
    }

    uint8_t wake_command[2] = {0x6B, 0x00};
    return i2c_master_transmit(mpu6050, wake_command, sizeof(wake_command), 100) == ESP_OK;
}

static void init_gpio(void)
{
    gpio_config_t output_config = {
        .pin_bit_mask = (1ULL << TRIG_LEFT) |
                        (1ULL << TRIG_RIGHT) |
                        (1ULL << MOTOR_1_DIR) |
                        (1ULL << MOTOR_2_DIR) |
                        (1ULL << MOTORS_ENABLE),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&output_config);

    gpio_config_t input_config = {
        .pin_bit_mask = (1ULL << ECHO_LEFT) | (1ULL << ECHO_RIGHT),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&input_config);

    gpio_set_level(TRIG_LEFT, 0);
    gpio_set_level(TRIG_RIGHT, 0);
    gpio_set_level(MOTORS_ENABLE, 1);
}

static void init_step_generator(void)
{
    ledc_timer_config_t timer_config = {
        .speed_mode = STEP_MODE,
        .duty_resolution = LEDC_TIMER_10_BIT,
        .timer_num = STEP_TIMER,
        .freq_hz = STEP_FREQUENCY_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer_config);

    ledc_channel_config_t motor_1_config = {
        .gpio_num = MOTOR_1_STEP,
        .speed_mode = STEP_MODE,
        .channel = MOTOR_1_CHANNEL,
        .intr_type = LEDC_INTR_DISABLE,
        .timer_sel = STEP_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    ledc_channel_config(&motor_1_config);

    ledc_channel_config_t motor_2_config = motor_1_config;
    motor_2_config.gpio_num = MOTOR_2_STEP;
    motor_2_config.channel = MOTOR_2_CHANNEL;
    ledc_channel_config(&motor_2_config);

    stop_motors();
}

static void init_uart(void)
{
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };

    uart_param_config(UART_NUM_0, &uart_config);
    uart_set_pin(UART_NUM_0, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE,
                 UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    uart_driver_install(UART_NUM_0, 256, 0, 0, NULL, 0);
}

static void control_task(void *argument)
{
    int64_t last_telemetry = 0;

    while (true) {
        distance_left = read_distance(TRIG_LEFT, ECHO_LEFT);
        vTaskDelay(pdMS_TO_TICKS(20));
        distance_right = read_distance(TRIG_RIGHT, ECHO_RIGHT);

        bool obstacle = distance_left <= STOP_DISTANCE_CM ||
                        distance_right <= STOP_DISTANCE_CM;

        // Передние датчики не запрещают отъехать назад от препятствия.
        if (obstacle && motion != 'S' && motion != 'B') {
            safe_stop = true;
            stop_motors();
            printf("STATE,SAFE_STOP,obstacle\n");
        }

        read_uart_command();

        int64_t now = esp_timer_get_time();
        if (now - last_telemetry >= 500000) {
            last_telemetry = now;
            printf("TELEMETRY,motion=%c,left_cm=%.1f,right_cm=%.1f,"
                   "safe_stop=%d,mpu6050=%s\n",
                   motion,
                   distance_left,
                   distance_right,
                   safe_stop ? 1 : 0,
                   mpu_found ? "OK" : "ERROR");
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void app_main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);

    init_gpio();
    init_step_generator();
    init_uart();
    mpu_found = start_mpu6050();

    printf("ESP32-S3 FreeRTOS готов к работе\n");
    printf("Команды: F-forward, B-back, L-left, R-right, S-stop/reset\n");

    xTaskCreate(control_task, "control_task", 4096, NULL, 5, NULL);
}
