import pywinusb.hid as hid

# Define the device VID and PID for the Elan WBF fingerprint scanner
VID = 0x04F3
PID = 0x0C02

# Find the device
devices = hid.HidDeviceFilter(vendor_id=VID, product_id=PID).get_devices()

if devices:
    device = devices[0]
    device.open()

    def enroll_fingerprint():
        print("Place your finger on the scanner...")
        # Send a command to the device to start the enrollment process
        device.send_feature_report([0x01, 0x00, 0x00, 0x00])
        # Read the response from the device
        response = device.read(64)
        # Process the response to extract the fingerprint data
        # ... (this part is omitted for brevity)
        print("Fingerprint enrolled successfully!")

    def verify_fingerprint():
        print("Place your finger on the scanner...")
        # Send a command to the device to start the verification process
        device.send_feature_report([0x02, 0x00, 0x00, 0x00])
        # Read the response from the device
        response = device.read(64)
        # Process the response to extract the verification result
        # ... (this part is omitted for brevity)
        print("Fingerprint verification result:", response)

    # Main program loop
    while True:
        print("1. Enroll a new fingerprint")
        print("2. Verify a fingerprint")
        print("3. Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            enroll_fingerprint()
        elif choice == "2":
            verify_fingerprint()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")

    # Close the device
    device.close()
else:
    print("No device found.")