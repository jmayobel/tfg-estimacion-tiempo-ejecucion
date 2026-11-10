// ============================================================================
// Azure Bicep Template: Dataset Creator VM Deployment
// ============================================================================
// Deploys Ubuntu 22.04 LTS VM with Docker support for dataset collection
// Supports both CPU-only and GPU-enabled (NC-series) VMs
// ============================================================================

@description('Name of the virtual machine')
param vmName string = 'dataset-creator-vm'

@description('Azure region for resource deployment')
param location string = resourceGroup().location

@description('VM size. The list is curated to fit the 6-vCPU regional cap of the Azure for Students subscription. All NC* GPU SKUs are at quota 0 by default and require a quota request before deploying. NCv1 (K80) is deliberately excluded (retired SKU, incompatible with nvidia-driver-535).')
@allowed([
  // Intel CPU (fits Total Regional vCPUs = 6)
  'Standard_D2s_v3'        // 2 vCPU, 8 GB RAM - low core-count point (decouples cores)
  'Standard_D4s_v3'        // 4 vCPU, 16 GB RAM - recommended CPU default
  // Intel memory-optimized — same cores as D4s_v3 but 2x RAM (decouples RAM).
  'Standard_E4s_v3'        // 4 vCPU, 32 GB RAM
  'Standard_E2s_v3'        // 2 vCPU, 16 GB RAM
  // AMD CPU (alternate architecture for the dataset)
  'Standard_D4as_v4'       // 4 vCPU, 16 GB RAM, AMD EPYC (Azure backs DASv4 with EPYC 7763/Milan)
  'Standard_D4ds_v4'       // 4 vCPU, 16 GB RAM + local NVMe (better FIO numbers)
  // ARM CPU — adds the arm64-on-Linux axis to the dataset.
  // v5 = Ampere Altra (Neoverse N1); v6 = Azure Cobalt 100 (Neoverse N2, newer/faster).
  // On the personal PAYG sub the Dpsv6 family already has quota (10 vCPU in
  // swedencentral) so v6 needs NO family quota request; v5 starts at 0/0.
  'Standard_D4ps_v5'       // 4 vCPU, 16 GB RAM, Ampere Altra
  'Standard_D8ps_v5'       // 8 vCPU, 32 GB RAM, Ampere Altra
  'Standard_D4ps_v6'       // 4 vCPU, 16 GB RAM, Cobalt 100 (clean 4-core ARM point)
  'Standard_D8ps_v6'       // 8 vCPU, 32 GB RAM, Cobalt 100 (also breaks the ~16GB RAM plateau)
  // Tesla T4 GPU - quota request required (NCASv3_T4 family)
  'Standard_NC4as_T4_v3'   // 4 vCPU, 28 GB RAM, Tesla T4 (recommended GPU default)
  'Standard_NC8as_T4_v3'   // 8 vCPU, 56 GB RAM, Tesla T4 (exceeds student quota)
  'Standard_NC16as_T4_v3'  // 16 vCPU, 110 GB RAM, Tesla T4 (exceeds student quota)
  // Tesla V100 GPU - quota request required (NCSv3 family); rarely granted to students
  'Standard_NC6s_v3'       // 6 vCPU, 112 GB RAM, Tesla V100
  'Standard_NC12s_v3'      // 12 vCPU, 224 GB RAM, Tesla V100 (exceeds student quota)
  'Standard_NC24s_v3'      // 24 vCPU, 448 GB RAM, Tesla V100 (exceeds student quota)
])
param vmSize string = 'Standard_D4s_v3'

@description('CIDR allowed to reach SSH (port 22). Defaults to public (0.0.0.0/0) — set to your office/home IP for production.')
param sshSourceAddressPrefix string = '*'

@description('CIDR allowed to reach Prometheus (port 9090). Defaults to VNet-only to avoid leaking metrics to the public internet.')
param prometheusSourceAddressPrefix string = 'VirtualNetwork'

@description('Expose Prometheus (port 9090) through the NSG at all. Disabled by default; rely on SSH tunnels for remote access.')
param exposePrometheus bool = false

@description('Admin username for SSH access')
param adminUsername string = 'azureuser'

@description('SSH public key for authentication')
@secure()
param sshPublicKey string

@description('OS disk type (Premium_LRS for SSD, StandardSSD_LRS for standard SSD)')
@allowed([
  'Premium_LRS'
  'StandardSSD_LRS'
  'Standard_LRS'
])
param osDiskType string = 'StandardSSD_LRS'

@description('OS disk size in GB')
param osDiskSizeGB int = 128

@description('Enable data disk for additional storage')
param enableDataDisk bool = false

@description('Data disk size in GB (for dataset storage)')
param dataDiskSizeGB int = 512

@description('Enable accelerated networking (requires compatible VM size)')
param enableAcceleratedNetworking bool = true

@description('Environment tag')
param environment string = 'production'

@description('Project tag')
param projectTag string = 'dataset-creator'

@description('Enable Spot VM for reduced costs')
param useSpot bool = false

// ============================================================================
// Variables
// ============================================================================

var nicName = '${vmName}-nic'
var nsgName = '${vmName}-nsg'
var publicIPName = '${vmName}-pip'
var vnetName = '${vmName}-vnet'
var subnetName = 'default'
var dataDiskName = '${vmName}-data-disk'

// Detect if this is a GPU VM based on SKU name
var isGpuVM = contains(toLower(vmSize), 'nc')

// Detect ARM (Ampere Altra / Cobalt) VMs — the Dpsv5/Dpsv6 families contain
// 'ps_v'. ARM VMs require an Arm64 OS image; x64 images fail at create time
// (NOT caught by what-if). Pick the matching Ubuntu Jammy image accordingly.
var isArmVM = contains(toLower(vmSize), 'ps_v')
var ubuntuImageSku = isArmVM ? '22_04-lts-arm64' : '22_04-lts-gen2'

// NCv1 (K80) SKUs do not support accelerated networking. Currently the @allowed list
// excludes them, but we keep this guard so adding them back doesn't silently break deploys.
var ncv1Sku = endsWith(toLower(vmSize), 'standard_nc6') || endsWith(toLower(vmSize), 'standard_nc12') || endsWith(toLower(vmSize), 'standard_nc24')
var effectiveAcceleratedNetworking = enableAcceleratedNetworking && !ncv1Sku

// ============================================================================
// Resources
// ============================================================================

// Network Security Group
resource nsg 'Microsoft.Network/networkSecurityGroups@2023-05-01' = {
  name: nsgName
  location: location
  properties: {
    securityRules: concat([
      {
        name: 'SSH'
        properties: {
          priority: 1000
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: sshSourceAddressPrefix
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '22'
          description: 'Allow SSH access from sshSourceAddressPrefix'
        }
      }
      {
        name: 'AllowOutbound'
        properties: {
          priority: 100
          protocol: '*'
          access: 'Allow'
          direction: 'Outbound'
          sourceAddressPrefix: '*'
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '*'
          description: 'Allow all outbound traffic'
        }
      }
    ], exposePrometheus ? [
      {
        name: 'Prometheus'
        properties: {
          priority: 1100
          protocol: 'Tcp'
          access: 'Allow'
          direction: 'Inbound'
          sourceAddressPrefix: prometheusSourceAddressPrefix
          sourcePortRange: '*'
          destinationAddressPrefix: '*'
          destinationPortRange: '9090'
          description: 'Allow Prometheus HTTP access from prometheusSourceAddressPrefix'
        }
      }
    ] : [])
  }
  tags: {
    environment: environment
    project: projectTag
  }
}

// Public IP Address
resource publicIP 'Microsoft.Network/publicIPAddresses@2023-05-01' = {
  name: publicIPName
  location: location
  sku: {
    name: 'Standard'
  }
  properties: {
    publicIPAllocationMethod: 'Static'
    dnsSettings: {
      domainNameLabel: toLower(vmName)
    }
  }
  tags: {
    environment: environment
    project: projectTag
  }
}

// Virtual Network
resource vnet 'Microsoft.Network/virtualNetworks@2023-05-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        '10.0.0.0/16'
      ]
    }
    subnets: [
      {
        name: subnetName
        properties: {
          addressPrefix: '10.0.0.0/24'
          networkSecurityGroup: {
            id: nsg.id
          }
        }
      }
    ]
  }
  tags: {
    environment: environment
    project: projectTag
  }
}

// Network Interface
resource nic 'Microsoft.Network/networkInterfaces@2023-05-01' = {
  name: nicName
  location: location
  properties: {
    ipConfigurations: [
      {
        name: 'ipconfig1'
        properties: {
          subnet: {
            id: vnet.properties.subnets[0].id
          }
          privateIPAllocationMethod: 'Dynamic'
          publicIPAddress: {
            id: publicIP.id
          }
        }
      }
    ]
    enableAcceleratedNetworking: effectiveAcceleratedNetworking
  }
  tags: {
    environment: environment
    project: projectTag
  }
}

// Data Disk (optional)
resource dataDisk 'Microsoft.Compute/disks@2023-04-02' = if (enableDataDisk) {
  name: dataDiskName
  location: location
  sku: {
    name: 'StandardSSD_LRS'
  }
  properties: {
    creationData: {
      createOption: 'Empty'
    }
    diskSizeGB: dataDiskSizeGB
  }
  tags: {
    environment: environment
    project: projectTag
  }
}

// Virtual Machine
resource vm 'Microsoft.Compute/virtualMachines@2023-09-01' = {
  name: vmName
  location: location
  properties: {
    hardwareProfile: {
      vmSize: vmSize
    }
    priority: useSpot ? 'Spot' : 'Regular'
    evictionPolicy: useSpot ? 'Deallocate' : null
    billingProfile: useSpot ? {
      maxPrice: -1
    } : null
    osProfile: {
      computerName: vmName
      adminUsername: adminUsername
      linuxConfiguration: {
        disablePasswordAuthentication: true
        ssh: {
          publicKeys: [
            {
              path: '/home/${adminUsername}/.ssh/authorized_keys'
              keyData: sshPublicKey
            }
          ]
        }
      }
    }
    storageProfile: {
      imageReference: {
        publisher: 'Canonical'
        offer: '0001-com-ubuntu-server-jammy'
        sku: ubuntuImageSku
        version: 'latest'
      }
      osDisk: {
        name: '${vmName}-osdisk'
        createOption: 'FromImage'
        managedDisk: {
          storageAccountType: osDiskType
        }
        diskSizeGB: osDiskSizeGB
      }
      dataDisks: enableDataDisk ? [
        {
          lun: 0
          name: dataDiskName
          createOption: 'Attach'
          managedDisk: {
            id: dataDisk.id
          }
        }
      ] : []
    }
    networkProfile: {
      networkInterfaces: [
        {
          id: nic.id
        }
      ]
    }
  }
  tags: {
    environment: environment
    project: projectTag
    gpu: isGpuVM ? 'true' : 'false'
    vmSize: vmSize
  }
}

// ============================================================================
// Outputs
// ============================================================================

@description('Public IP address of the VM')
output publicIP string = publicIP.properties.ipAddress

@description('FQDN of the VM')
output fqdn string = publicIP.properties.dnsSettings.fqdn

@description('SSH connection string')
output sshCommand string = 'ssh ${adminUsername}@${publicIP.properties.ipAddress}'

@description('VM resource ID')
output vmId string = vm.id

@description('Is this a GPU-enabled VM')
output isGpuVM bool = isGpuVM

@description('VM size')
output vmSize string = vmSize
