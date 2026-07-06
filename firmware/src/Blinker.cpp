
#define LED_PIN 21
#define NUM_LEDS 1

#define BLINK_ON 200
#define BLINK_OFF 200
#define DIGIT_PAUSE 800
#define ZERO_LONG 1200

#if 0

Adafruit_NeoPixel led(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

void led_on()
{
    led.setPixelColor(0, led.Color(50,50,50));
    led.show();
}

void led_off()
{
    led.clear();
    led.show();
}

void blink_once()
{
    led_on();
    delay(BLINK_ON);
    led_off();
    delay(BLINK_OFF);
}

void blink_digit(int digit)
{
    if(digit == 0)
    {
        led_on();
        delay(ZERO_LONG);
        led_off();
    }
    else
    {
        for(int i=0;i<digit;i++)
        {
            blink_once();
        }
    }

    delay(DIGIT_PAUSE);
}

#endif

void blink_number(int number)
{
    /*
    char buffer[12];
    sprintf(buffer,"%d",number);

    for(int i=0; buffer[i] != '\0'; i++)
    {
        int digit = buffer[i] - '0';
        blink_digit(digit);
    }
    */
}
